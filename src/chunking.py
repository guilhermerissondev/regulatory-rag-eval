"""Segmentação do texto normativo.

Chunking por janela fixa é o padrão em tutorial de RAG e é a escolha
errada aqui, por dois motivos.

O primeiro é de qualidade: texto normativo tem unidade semântica
declarada. Um artigo com seus parágrafos e incisos é uma ideia completa;
cortá-lo a cada 1.000 caracteres separa a regra da sua exceção, e a
exceção recuperada sozinha diz o oposto da norma.

O segundo é o que decidiu a arquitetura deste projeto: **segmentar por
artigo é o que torna a avaliação possível**. Se o chunk carrega a
etiqueta do artigo, cada citação da resposta pode ser cruzada com o
estado daquele artigo. Com janela fixa, essa checagem não existe.

Uma premissa foi verificada e derrubada no caminho, e vale registrar
porque mudou o desenho: o BACEN publica **texto consolidado**, não a
redação original. Trechos alterados aparecem já com a redação nova e um
marcador de atribuição — "(Redação dada, a partir de 2/8/2021, pela
Resolução BCB nº 118)". Ou seja, o risco NÃO é o sistema citar redação
velha, como se supunha no início.

O risco real, medido no corpus, é outro e tem duas faces:

    norma revogada por inteiro   131 dos 294 normativos (44,6%)
    artigo revogado dentro de    151 chunks marcados com "(Revogado...)"
    norma ainda vigente          no próprio texto

O segundo caso é o mais traiçoeiro: a norma está em vigor, o texto do
artigo continua no documento, e ele não vale mais.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import pandas as pd

from . import config
from .corpus import Normativo, artigos_alterados

# "Art. 1º", "Art. 31-A", "Art. 105." — o sufixo com letra aparece em
# artigos inseridos por alteração posterior, e é justamente onde mora a
# maior parte das mudanças.
# O marcador ordinal vem ANTES do sufixo: o texto escreve "Art. 3º-A", não
# "Art. 3-Aº". Trocar essa ordem na regex fazia todo artigo com sufixo ser
# lido como o artigo-base — e são justamente os inseridos por alteração
# posterior, ou seja, os que mais interessam à checagem de vigência.
_RE_ARTIGO_INICIO = re.compile(
    r"(?:^|\n)\s*(Art\.\s*(\d+)\s*[º°]?(?:\s*-\s*([A-Z]))?)", re.I
)
_RE_CAPITULO = re.compile(r"(?:^|\n)\s*(CAP[ÍI]TULO\s+[IVXLC]+[^\n]{0,80})", re.I)
_RE_SECAO = re.compile(r"(?:^|\n)\s*(Se[çc][ãa]o\s+[IVXLC]+[^\n]{0,80})", re.I)

# Marcadores de revisão que o BACEN embute no texto consolidado.
_RE_REVOGADO_INLINE = re.compile(r"\(\s*Revogad[oa]\b", re.I)
_RE_NOVA_REDACAO = re.compile(r"\(\s*(?:Reda[çc][ãa]o dada|Inclu[íi]d[oa])\b", re.I)

# Uma resolução que aprova regulamento anexo tem DUAS numerações de artigo
# rodando em paralelo: a da resolução (curta, "Fica aprovado o Regulamento
# anexo") e a do regulamento (longa, onde mora a regra de verdade). Na
# Resolução BCB nº 1 existem dois "Art. 1º", a 20 mil caracteres um do
# outro. Sem separar as partes, "art. 3" é ambíguo e a auditoria de
# citação passa a comparar coisas diferentes achando que são a mesma.
_RE_ANEXO = re.compile(
    r"(?:^|\n)\s*((?:REGULAMENTO\s+ANEXO|ANEXO\s+[IVX0-9]*)[^\n]{0,90})"
)


@dataclass
class Chunk:
    id: str
    normativo_id: str
    rotulo_norma: str
    tipo: str
    numero: str
    artigo: str | None
    # Parte do documento: vazio para o corpo da norma, rótulo do anexo
    # quando o artigo pertence a regulamento anexo. Desambigua numeração.
    parte: str
    capitulo: str
    secao: str
    texto: str
    # A norma inteira foi revogada.
    revogado: bool
    # O artigo foi revogado por norma posterior, mas segue impresso no
    # documento com marcador. A norma continua válida; este artigo não.
    artigo_revogado: bool
    # O artigo teve nova redação ou foi incluído depois. NÃO é erro citá-lo:
    # o texto exibido já é o vigente. Serve para saber que é volátil.
    artigo_alterado: bool
    data: str

    @property
    def citacao(self) -> str:
        """Como o trecho deve ser citado na resposta."""
        base = self.rotulo_norma
        if self.parte:
            base += f", {self.parte}"
        if self.artigo:
            return f"{base}, art. {self.artigo}"
        return base


def _hierarquia(texto: str) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    caps = [(m.start(), m.group(1).strip()) for m in _RE_CAPITULO.finditer(texto)]
    secs = [(m.start(), m.group(1).strip()) for m in _RE_SECAO.finditer(texto)]
    return caps, secs


def _partes(texto: str) -> list[tuple[int, str]]:
    """Fronteiras de anexo, com rótulo curto para compor a citação."""
    marcos: list[tuple[int, str]] = []
    for m in _RE_ANEXO.finditer(texto):
        bruto = m.group(1).strip()
        rotulo = "Regulamento anexo" if re.match(r"REGULAMENTO", bruto, re.I) else "Anexo"
        num = re.search(r"ANEXO\s+([IVX]+|\d+)", bruto, re.I)
        if num and rotulo == "Anexo":
            rotulo = f"Anexo {num.group(1).upper()}"
        marcos.append((m.start(), rotulo))
    return marcos


def _vigente_em(marcos: list[tuple[int, str]], pos: int) -> str:
    """Último marco de hierarquia antes desta posição."""
    atual = ""
    for p, rotulo in marcos:
        if p <= pos:
            atual = rotulo
        else:
            break
    return atual


def _dividir_longo(texto: str, maximo: int, overlap: int) -> list[str]:
    """Quebra um artigo grande demais, preferindo fronteira de parágrafo.

    Mesmo quebrado, todos os pedaços mantêm a etiqueta do artigo — a
    rastreabilidade não pode se perder por causa do tamanho.
    """
    if len(texto) <= maximo:
        return [texto]

    partes, inicio = [], 0
    while inicio < len(texto):
        fim = min(inicio + maximo, len(texto))
        if fim < len(texto):
            corte = texto.rfind("\n", inicio + maximo // 2, fim)
            if corte == -1:
                corte = texto.rfind(". ", inicio + maximo // 2, fim)
            if corte != -1:
                fim = corte + 1
        partes.append(texto[inicio:fim].strip())
        if fim >= len(texto):
            break
        inicio = max(fim - overlap, inicio + 1)
    return [p for p in partes if p]


def por_artigo(n: Normativo) -> list[Chunk]:
    """Segmenta respeitando a estrutura do normativo."""
    texto = n.texto
    alterados = artigos_alterados(n)
    caps, secs = _hierarquia(texto)
    partes = _partes(texto)

    marcos = [(m.start(), m.group(2), m.group(3)) for m in _RE_ARTIGO_INICIO.finditer(texto)]
    chunks: list[Chunk] = []

    def adicionar(trecho: str, artigo: str | None, pos: int) -> None:
        trecho = trecho.strip()
        # Piso de tamanho para descartar fragmento solto entre marcadores
        # ("Seção II" sozinho, por exemplo). Um artigo de verdade escapa do
        # piso mesmo curto: "Art. 4º Revogado." é curto e é exatamente o
        # tipo de trecho que alguém vai querer recuperar.
        minimo = 15 if artigo else 60
        if len(trecho) < minimo:
            return
        rotulo_parte = _vigente_em(partes, pos)
        for j, parte in enumerate(_dividir_longo(trecho, config.MAX_CHARS_CHUNK, config.OVERLAP_CHARS)):
            chunks.append(
                Chunk(
                    id=f"{n.id}::{rotulo_parte or 'corpo'}::{artigo or 'preambulo'}::{j}",
                    normativo_id=n.id,
                    rotulo_norma=n.rotulo,
                    tipo=n.tipo,
                    numero=n.numero,
                    artigo=artigo,
                    parte=rotulo_parte,
                    capitulo=_vigente_em(caps, pos),
                    secao=_vigente_em(secs, pos),
                    texto=parte,
                    revogado=n.revogado,
                    artigo_revogado=bool(_RE_REVOGADO_INLINE.search(parte)),
                    artigo_alterado=bool(
                        _RE_NOVA_REDACAO.search(parte) or (artigo and artigo in alterados)
                    ),
                    data=n.data,
                )
            )

    if not marcos:
        adicionar(texto, None, 0)
        return chunks

    # Preâmbulo: ementa, fundamento legal e o "R E S O L V E".
    adicionar(texto[: marcos[0][0]], None, 0)

    for i, (pos, num, sufixo) in enumerate(marcos):
        fim = marcos[i + 1][0] if i + 1 < len(marcos) else len(texto)
        artigo = f"{num}-{sufixo.upper()}" if sufixo else num
        adicionar(texto[pos:fim], artigo, pos)

    return chunks


def janela_fixa(n: Normativo, tamanho: int = None, overlap: int = None) -> list[Chunk]:
    """Baseline: corta a cada N caracteres, ignorando a estrutura.

    Existe para ser comparado, não para ser usado. Repare que `artigo`
    fica sempre nulo — é isso que inviabiliza a checagem de vigência.
    """
    tamanho = config.MAX_CHARS_CHUNK if tamanho is None else tamanho
    overlap = config.OVERLAP_CHARS if overlap is None else overlap

    partes = _dividir_longo(n.texto, tamanho, overlap)
    return [
        Chunk(
            id=f"{n.id}::janela::{j}",
            normativo_id=n.id,
            rotulo_norma=n.rotulo,
            tipo=n.tipo,
            numero=n.numero,
            artigo=None,
            parte="",
            capitulo="",
            secao="",
            texto=parte,
            revogado=n.revogado,
            artigo_revogado=False,
            artigo_alterado=False,
            data=n.data,
        )
        for j, parte in enumerate(partes)
    ]


def construir(normativos: list[Normativo], estrategia: str = "artigo") -> pd.DataFrame:
    funcao = {"artigo": por_artigo, "janela": janela_fixa}[estrategia]
    linhas = [asdict(c) for n in normativos for c in funcao(n)]
    df = pd.DataFrame(linhas)
    df["n_chars"] = df["texto"].str.len()
    return df


def texto_para_embedding(df: pd.DataFrame) -> list[str]:
    """Contexto hierárquico prefixado ao trecho.

    Um artigo isolado costuma dizer "aplica-se o disposto no caput" sem
    nomear o assunto. Carregar norma, capítulo e seção no texto embutido
    dá ao vetor o contexto que o trecho sozinho não tem.
    """
    partes = []
    for r in df.itertuples():
        cabecalho = r.rotulo_norma
        if r.capitulo:
            cabecalho += f" — {r.capitulo}"
        if r.secao:
            cabecalho += f" — {r.secao}"
        if r.artigo:
            cabecalho += f" — art. {r.artigo}"
        partes.append(f"{config.PREFIXO_PASSAGEM}{cabecalho}\n{r.texto}")
    return partes
