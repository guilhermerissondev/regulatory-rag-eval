"""Ingestão dos normativos do BACEN.

Dois endpoints não documentados, descobertos pela aba de rede do site:

    buscanormativos   metadados e busca textual
    exibenormativo    texto integral, por tipo e número

O segundo devolve muito mais que o texto. Devolve o histórico de
alterações artigo por artigo e a marcação de revogação — que é
exatamente a verdade de referência que o harness precisa para checar se
o sistema está citando norma que não vale mais.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict, field

from . import config

_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json,*/*"}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def _get_json(url: str, tentativas: int = 3) -> dict:
    ultimo = None
    for i in range(tentativas):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            ultimo = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"falhou após {tentativas} tentativas: {url}") from ultimo


# --------------------------------------------------------------------------
# Limpeza
# --------------------------------------------------------------------------
def limpar_html(bruto: str | None) -> str:
    """HTML do SharePoint para texto corrido.

    Os normativos vêm de um CMS antigo: entidades duplamente escapadas,
    `<div class="ExternalClass...">` em volta de tudo e `\\r\\n` no meio
    das frases. Sem essa limpeza, o chunk carrega lixo que o embedding
    trata como conteúdo.
    """
    if not bruto:
        return ""
    txt = html.unescape(html.unescape(bruto))
    txt = re.sub(r"<br\s*/?>", "\n", txt, flags=re.I)
    txt = re.sub(r"</(p|div|tr|li)>", "\n", txt, flags=re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = txt.replace("\xa0", " ").replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r" *\n *", "\n", txt)  # tags removidas deixam espaço colado na quebra
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def _separar_lista(bruto: str | None) -> list[str]:
    """Os campos multivalorados usam `;#` como separador do SharePoint."""
    if not bruto:
        return []
    itens = [limpar_html(p).strip() for p in bruto.split(";#")]
    return [i for i in itens if i]


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
@dataclass
class Normativo:
    id: str
    tipo: str
    numero: str
    titulo: str
    assunto: str
    data: str
    revogado: bool
    cancelado: bool
    texto: str
    atualizacoes: list[str] = field(default_factory=list)
    referencias: list[str] = field(default_factory=list)
    normas_vinculadas: list[str] = field(default_factory=list)
    dou: str = ""

    @property
    def rotulo(self) -> str:
        """Como a norma é citada em texto: 'Resolução BCB nº 1'."""
        return f"{self.tipo} nº {self.numero}"


# Artigos alterados por normativo posterior. O campo `Atualizacoes` traz
# frases como "Nova redação: art. 6º, parágrafo único; art. 23, § 3º".
_RE_ARTIGO = re.compile(r"art\.\s*(\d+)\s*[º°]?(?:\s*-\s*([A-Z]))?", re.I)


def artigos_alterados(n: Normativo) -> set[str]:
    """Números de artigo que sofreram alteração posterior.

    Serve para o harness: se a resposta cita um desses artigos sem
    ressalvar que a redação mudou, é erro — mesmo que o trecho recuperado
    exista no texto original.
    """
    alterados: set[str] = set()
    for entrada in n.atualizacoes:
        for num, sufixo in _RE_ARTIGO.findall(entrada):
            alterados.add(f"{num}-{sufixo.upper()}" if sufixo else num)
    return alterados


# --------------------------------------------------------------------------
# Busca e download
# --------------------------------------------------------------------------
def buscar(consulta: str = None, limite: int = None) -> list[dict]:
    """Lista normativos que casam com a consulta.

    A API pagina por `startrow`; o total vem em `TotalRows`.
    """
    consulta = config.CONSULTA if consulta is None else consulta
    limite = config.MAX_NORMATIVOS if limite is None else limite

    filtro = f"ContentType:normativo AND contentSource:normativos AND {consulta}"
    achados: list[dict] = []
    inicio, pagina = 0, 50

    while len(achados) < limite:
        url = (
            f"{config.API_BUSCA}?querytext={urllib.parse.quote(filtro)}"
            f"&rowlimit={pagina}&startrow={inicio}"
        )
        d = _get_json(url)
        linhas = d.get("Rows", [])
        if not linhas:
            break
        achados.extend(linhas)
        inicio += pagina
        if inicio >= d.get("TotalRows", 0):
            break
        time.sleep(config.PAUSA_ENTRE_REQUISICOES)

    return achados[:limite]


def baixar(tipo: str, numero: str) -> Normativo | None:
    """Texto integral e metadados de um normativo."""
    url = (
        f"{config.API_TEXTO}?p1={urllib.parse.quote(tipo)}"
        f"&p2={urllib.parse.quote(str(numero))}"
    )
    try:
        d = _get_json(url)
    except RuntimeError:
        return None

    c = d.get("conteudo")
    if isinstance(c, list):
        c = c[0] if c else None
    if not isinstance(c, dict) or not c.get("Texto"):
        return None

    return Normativo(
        id=str(c.get("Id") or f"{tipo}-{numero}"),
        tipo=limpar_html(c.get("Tipo")) or tipo,
        numero=str(c.get("Numero") or numero).split(".")[0],
        titulo=limpar_html(c.get("Titulo")),
        assunto=limpar_html(c.get("Assunto")),
        data=str(c.get("Data") or "")[:10],
        revogado=bool(int(c.get("Revogado") or 0)),
        cancelado=bool(int(c.get("Cancelado") or 0)),
        texto=limpar_html(c.get("Texto")),
        atualizacoes=_separar_lista(c.get("Atualizacoes")),
        referencias=_separar_lista(c.get("Referencias")),
        normas_vinculadas=_separar_lista(c.get("NormasVinculadas")),
        dou=limpar_html(c.get("DOU")),
    )


def coletar(consulta: str = None, limite: int = None, verboso: bool = True) -> list[Normativo]:
    """Busca, baixa e guarda o corpus em JSONL."""
    metadados = buscar(consulta, limite)
    if verboso:
        print(f"  {len(metadados)} normativos encontrados na busca")

    vistos: set[tuple[str, str]] = set()
    saida: list[Normativo] = []

    for i, m in enumerate(metadados, 1):
        tipo = limpar_html(m.get("TipodoNormativoOWSCHCS") or "")
        numero = str(m.get("NumeroOWSNMBR") or "").split(".")[0]
        if not tipo or not numero or (tipo, numero) in vistos:
            continue
        vistos.add((tipo, numero))

        n = baixar(tipo, numero)
        if n and len(n.texto) > 400:
            saida.append(n)
        if verboso and i % 25 == 0:
            print(f"  {i}/{len(metadados)} processados | {len(saida)} com texto util")
        time.sleep(config.PAUSA_ENTRE_REQUISICOES)

    salvar(saida)
    return saida


def salvar(normativos: list[Normativo]) -> None:
    with config.CORPUS_JSONL.open("w", encoding="utf-8") as fh:
        for n in normativos:
            fh.write(json.dumps(asdict(n), ensure_ascii=False) + "\n")


def carregar() -> list[Normativo]:
    if not config.CORPUS_JSONL.exists():
        return []
    with config.CORPUS_JSONL.open(encoding="utf-8") as fh:
        return [Normativo(**json.loads(l)) for l in fh if l.strip()]


def resumo(normativos: list[Normativo]) -> dict:
    revogados = sum(n.revogado for n in normativos)
    com_alteracao = sum(1 for n in normativos if artigos_alterados(n))
    return {
        "normativos": len(normativos),
        "revogados": revogados,
        "pct_revogados": revogados / max(len(normativos), 1),
        "com_artigos_alterados": com_alteracao,
        "chars_totais": sum(len(n.texto) for n in normativos),
        "tipos": sorted({n.tipo for n in normativos}),
    }
