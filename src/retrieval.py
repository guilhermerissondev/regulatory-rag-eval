"""Recuperação: densa, léxica e híbrida.

Busca densa entende paráfrase ("quanto tempo tenho para contestar" acha
"prazo para abertura do MED") mas erra em termo técnico exato e em
número de artigo. BM25 faz o oposto: acerta "Resolução BCB nº 30" e não
faz ideia do que é sinônimo.

Em corpus normativo os dois casos aparecem o tempo todo, e é por isso
que a fusão costuma ganhar das duas isoladas. Este módulo implementa as
três para que o harness compare, em vez de escolher por fé.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


def _citacao(r) -> str:
    base = r["rotulo_norma"]
    if r.get("parte"):
        base += f", {r['parte']}"
    return f"{base}, art. {r['artigo']}" if r["artigo"] else base


@dataclass
class Resultado:
    posicao: int
    chunk_id: str
    score: float
    citacao: str
    texto: str
    artigo: str | None
    parte: str
    revogado: bool
    artigo_revogado: bool
    artigo_alterado: bool
    rotulo_norma: str


class Indice:
    """Índice de busca sobre os chunks.

    Guarda o dataframe e as duas representações — vetorial e léxica.
    """

    def __init__(self, chunks: pd.DataFrame, modelo=None, embeddings: np.ndarray = None):
        self.chunks = chunks.reset_index(drop=True)
        self._modelo = modelo
        self.embeddings = embeddings
        self._bm25 = None
        self._construir_bm25()

    # -- construção -------------------------------------------------------
    def _construir_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        self._tokens = [_tokenizar(t) for t in self.chunks["texto"]]
        self._bm25 = BM25Okapi(self._tokens)

    @property
    def modelo(self):
        """Carrega o encoder sob demanda: o import do torch é lento e nem
        toda operação precisa dele."""
        if self._modelo is None:
            from sentence_transformers import SentenceTransformer

            self._modelo = SentenceTransformer(config.MODELO_EMBEDDING)
        return self._modelo

    def indexar(self, textos: list[str], lote: int = 64, verboso: bool = True) -> np.ndarray:
        """Calcula e guarda os embeddings das passagens."""
        self.embeddings = self.modelo.encode(
            textos,
            batch_size=lote,
            show_progress_bar=verboso,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return self.embeddings

    # -- busca ------------------------------------------------------------
    def _montar(self, ordem: np.ndarray, scores: np.ndarray, k: int) -> list[Resultado]:
        saida = []
        for pos, i in enumerate(ordem[:k], start=1):
            r = self.chunks.iloc[i]
            saida.append(
                Resultado(
                    posicao=pos,
                    chunk_id=r["id"],
                    score=float(scores[i]),
                    citacao=_citacao(r),
                    texto=r["texto"],
                    artigo=r["artigo"],
                    parte=str(r.get("parte", "") or ""),
                    revogado=bool(r["revogado"]),
                    artigo_revogado=bool(r.get("artigo_revogado", False)),
                    artigo_alterado=bool(r["artigo_alterado"]),
                    rotulo_norma=r["rotulo_norma"],
                )
            )
        return saida

    def denso(self, pergunta: str, k: int = None) -> list[Resultado]:
        k = config.TOP_K if k is None else k
        if self.embeddings is None:
            raise RuntimeError("índice denso não construído; rode indexar()")
        # O E5 exige o prefixo "query:"; sem ele a recuperação piora sem
        # emitir erro nenhum, que é o tipo de bug que só aparece na métrica.
        v = self.modelo.encode(
            [config.PREFIXO_CONSULTA + pergunta],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        scores = self.embeddings @ v
        return self._montar(np.argsort(-scores), scores, k)

    def bm25(self, pergunta: str, k: int = None) -> list[Resultado]:
        k = config.TOP_K if k is None else k
        scores = np.asarray(self._bm25.get_scores(_tokenizar(pergunta)))
        return self._montar(np.argsort(-scores), scores, k)

    def hibrido(self, pergunta: str, k: int = None, peso_denso: float = None) -> list[Resultado]:
        """Fusão por reciprocal rank.

        A fusão é feita sobre POSIÇÃO, não sobre score. Similaridade de
        cosseno e score BM25 vivem em escalas diferentes e sem relação —
        somá-los diretamente deixa o resultado à mercê de qual dos dois
        por acaso produz números maiores.
        """
        k = config.TOP_K if k is None else k
        peso = config.PESO_DENSO if peso_denso is None else peso_denso
        c = 60  # amortecedor padrão do RRF

        fundido = np.zeros(len(self.chunks))
        for lista, w in ((self.denso(pergunta, k * 4), peso),
                         (self.bm25(pergunta, k * 4), 1 - peso)):
            for r in lista:
                i = self.chunks.index[self.chunks["id"] == r.chunk_id][0]
                fundido[i] += w / (c + r.posicao)

        return self._montar(np.argsort(-fundido), fundido, k)

    def buscar(self, pergunta: str, estrategia: str = "hibrido", k: int = None):
        return {"denso": self.denso, "bm25": self.bm25, "hibrido": self.hibrido}[estrategia](
            pergunta, k
        )


def _tokenizar(texto: str) -> list[str]:
    """Tokenização simples com dobra de acento.

    Consulta de usuário raramente vem acentuada corretamente; o texto
    normativo sempre vem. Sem normalizar, o BM25 perde casamento óbvio.
    """
    import unicodedata

    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return [p for p in _RE_TOKEN.findall(t) if len(p) > 1]


import re  # noqa: E402

_RE_TOKEN = re.compile(r"[a-z0-9º°]+")
