"""Resolução de versões empilhadas.

Este módulo existe por causa de um achado que só apareceu ao ler o texto
bruto, e que redefiniu o projeto.

O BACEN publica texto consolidado, mas consolidado de um jeito
particular: **todas as versões históricas de cada dispositivo ficam
empilhadas em sequência**, e só a última vale. O art. 39 do Regulamento
do Pix é o exemplo canônico:

    II - houver problemas na identificação do usuário recebedor.
    II - ... ; ou   (Redação dada, a partir de 1º/2/2022, pela Res. BCB nº 181)
    II - ... ;      (Redação dada pela Res. BCB nº 402, de 22/7/2024.)

Três gerações do mesmo inciso, coladas, sem separador. No corpus deste
projeto isso atinge **1.128 dos 6.360 chunks (17,7%)**, com até sete
versões do mesmo dispositivo num único trecho.

Para um sistema de RAG isso é veneno. O trecho recuperado é literalmente
correto — está lá, no documento oficial — e ainda assim contém texto que
não vale mais. O modelo não tem como distinguir, e a tendência é citar a
primeira ocorrência, que é justamente a mais antiga.

A regra de desempate é simples e é a que o próprio documento impõe:
**dentro de um artigo, a última ocorrência de um dispositivo é a
vigente.**
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Rótulos de dispositivo. Incisos em romano maiúsculo seguidos de travessão,
# parágrafos numerados e o parágrafo único.
_RE_LABEL = re.compile(
    r"(?:(?<=\s)|^)((?:[IVXLC]{1,7}\s*[-–]|§\s*\d+\s*[º°]?|Parágrafo único\s*[-–.]?))"
)

# Marcadores de atribuição que o BACEN anexa à versão nova.
_RE_ATRIBUICAO = re.compile(
    r"\((?:Reda[çc][ãa]o dada|Inclu[íi]d[oa]|Revogad[oa])[^)]{0,220}\)", re.I
)
_RE_REVOGADO = re.compile(r"\(\s*Revogad[oa]\b", re.I)


@dataclass
class Dispositivo:
    rotulo: str
    texto: str
    tem_atribuicao: bool
    revogado: bool
    posicao: int


def _normalizar_rotulo(bruto: str) -> str:
    r = re.sub(r"\s+", "", bruto).rstrip("-–.")
    return r.upper()


def segmentar(texto: str) -> tuple[str, list[Dispositivo]]:
    """Separa o caput dos dispositivos numerados.

    Devolve (caput, dispositivos) preservando a ordem original.
    """
    marcos = list(_RE_LABEL.finditer(texto))
    if not marcos:
        return texto, []

    caput = texto[: marcos[0].start()]
    disp: list[Dispositivo] = []

    for i, m in enumerate(marcos):
        fim = marcos[i + 1].start() if i + 1 < len(marcos) else len(texto)
        trecho = texto[m.start():fim].strip()
        disp.append(
            Dispositivo(
                rotulo=_normalizar_rotulo(m.group(1)),
                texto=trecho,
                tem_atribuicao=bool(_RE_ATRIBUICAO.search(trecho)),
                revogado=bool(_RE_REVOGADO.search(trecho)),
                posicao=m.start(),
            )
        )
    return caput, disp


def tem_empilhamento(texto: str) -> bool:
    """Há dispositivo repetido com marcador de atribuição em pelo menos
    uma das ocorrências?

    A checagem exige a atribuição de propósito. Rótulo repetido sozinho
    pode ser lista aninhada legítima; rótulo repetido onde uma das cópias
    diz "(Redação dada...)" é empilhamento de versão.
    """
    _, disp = segmentar(texto)
    por_rotulo: dict[str, list[Dispositivo]] = {}
    for d in disp:
        por_rotulo.setdefault(d.rotulo, []).append(d)
    return any(
        len(v) > 1 and any(x.tem_atribuicao for x in v) for v in por_rotulo.values()
    )


def resolver(texto: str, manter_revogados: bool = False) -> str:
    """Devolve o texto com apenas a versão vigente de cada dispositivo.

    Regras, nesta ordem:

    1. Dentro do artigo, a ÚLTIMA ocorrência de um rótulo é a vigente.
       É a convenção do próprio documento: cada alteração é anexada
       depois da anterior.
    2. Se a versão vigente está marcada como revogada, o dispositivo sai
       do texto — ele não existe mais, e mantê-lo é o erro que este
       módulo veio corrigir.
    3. Rótulo que aparece uma vez só passa intacto.

    A ordem de saída segue a posição da versão vigente, o que reproduz a
    sequência natural (I, II, III...) porque as versões de um mesmo
    dispositivo ficam agrupadas.
    """
    caput, disp = segmentar(texto)
    if not disp:
        return texto.strip()

    ultimos: dict[str, Dispositivo] = {}
    for d in disp:
        ultimos[d.rotulo] = d  # sobrescreve: fica a última

    vigentes = [d for d in ultimos.values() if manter_revogados or not d.revogado]
    vigentes.sort(key=lambda d: d.posicao)

    partes = [caput.strip()] + [d.texto for d in vigentes]
    return re.sub(r"[ \t]+", " ", "\n".join(p for p in partes if p)).strip()


def limpar_atribuicoes(texto: str) -> str:
    """Remove os parentéticos de atribuição.

    Só deve rodar DEPOIS de `resolver`: a atribuição é o que identifica
    qual versão é mais nova. Tirá-la antes destrói a informação que
    permite desempatar.
    """
    return re.sub(r"\s+", " ", _RE_ATRIBUICAO.sub("", texto)).strip()


def estatisticas(textos: list[str]) -> dict:
    """Quanto o corpus encolhe ao ficar só com a redação vigente."""
    total, afetados, chars_antes, chars_depois = len(textos), 0, 0, 0
    for t in textos:
        r = resolver(t)
        chars_antes += len(t)
        chars_depois += len(r)
        if tem_empilhamento(t):
            afetados += 1
    return {
        "chunks": total,
        "com_empilhamento": afetados,
        "pct_afetados": afetados / max(total, 1),
        "chars_antes": chars_antes,
        "chars_depois": chars_depois,
        "reducao": 1 - chars_depois / max(chars_antes, 1),
    }
