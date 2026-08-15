"""Métricas de recuperação e de vigência.

A parte de recuperação é convencional: recall@k, MRR, precisão@k. O que
não é convencional está no fim do arquivo — as métricas de vigência, que
medem se o sistema respondeu com norma que não vale mais.

Nenhuma delas precisa de LLM. Metade de um harness de RAG é avaliável
sem gastar um token, e é a metade que a maioria dos projetos pula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------
# Julgamento de relevância
# --------------------------------------------------------------------------
def _chave(norma: str, artigo: str | None, parte: str = "") -> str:
    """Identidade de um dispositivo.

    A parte entra na chave porque numeração de artigo se repete entre o
    corpo da norma e seu regulamento anexo. Sem isso, "art. 3" casa com
    dois textos diferentes e a auditoria compara coisas distintas.
    """
    return f"{_normalizar(norma)}|{_normalizar(parte)}|{artigo or ''}"


def _normalizar(s: str) -> str:
    """'Resolução BCB N° 1' e 'Resolução BCB nº 1' são a mesma norma.

    A ordem das operações importa e já esteve errada aqui. O NFKD
    decompõe o ordinal masculino "º" em "o" mas deixa o sinal de grau
    "°" intacto — então normalizar primeiro fazia "nº 1" virar "no 1" e
    "n° 1" virar "n 1", e as duas grafias deixavam de casar.

    O efeito seria silencioso e caro: citação legítima do modelo marcada
    como inexistente, inflando a taxa de alucinação do relatório.
    """
    import re
    import unicodedata

    t = str(s).lower()
    for marcador in ("n°", "nº", "n.º", "no."):
        t = t.replace(marcador, "n")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def relevantes_da_pergunta(item: dict) -> set[str]:
    return {
        _chave(a["norma"], a.get("artigo"), a.get("parte", ""))
        for a in item.get("artigos_relevantes", [])
    }


def marcar_relevancia(resultados, item: dict) -> list[bool]:
    alvo = relevantes_da_pergunta(item)
    return [_chave(r.rotulo_norma, r.artigo, r.parte) in alvo for r in resultados]


# --------------------------------------------------------------------------
# Recuperação
# --------------------------------------------------------------------------
def recall_em_k(marcas: list[bool], total_relevantes: int, k: int) -> float:
    """Fração dos trechos-alvo que apareceram nos k primeiros.

    O denominador é o número de artigos que a pergunta exige, não o
    número de acertos possíveis — se a resposta correta depende de dois
    artigos e só um foi recuperado, o recall é 0,5 mesmo que o outro não
    exista no corpus.
    """
    if total_relevantes == 0:
        return float("nan")
    return sum(marcas[:k]) / total_relevantes


def precisao_em_k(marcas: list[bool], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(marcas[:k]) / k


def rr(marcas: list[bool]) -> float:
    """Recíproco da posição do primeiro acerto."""
    for i, m in enumerate(marcas, start=1):
        if m:
            return 1.0 / i
    return 0.0


def avaliar_recuperacao(indice, golden: list[dict], estrategia: str, k_max: int = None) -> pd.DataFrame:
    """Roda o conjunto dourado inteiro e devolve uma linha por pergunta."""
    k_max = max(config.KS_AVALIADOS) if k_max is None else k_max
    linhas = []

    for item in golden:
        resultados = indice.buscar(item["pergunta"], estrategia=estrategia, k=k_max)
        marcas = marcar_relevancia(resultados, item)
        n_rel = len(relevantes_da_pergunta(item))

        linha = {
            "id": item["id"],
            "categoria": item.get("categoria", ""),
            "armadilha": item.get("armadilha") or "",
            "estrategia": estrategia,
            "relevantes": n_rel,
            "mrr": rr(marcas),
            "primeiro_acerto": next((i for i, m in enumerate(marcas, 1) if m), None),
        }
        for k in config.KS_AVALIADOS:
            linha[f"recall@{k}"] = recall_em_k(marcas, n_rel, k)
            linha[f"precisao@{k}"] = precisao_em_k(marcas, k)

        # Contaminação do contexto: o que foi entregue ao gerador já vem
        # com norma revogada ou artigo alterado dentro?
        topo = resultados[: config.TOP_K]
        linha["ctx_revogado"] = sum(r.revogado for r in topo)
        linha["ctx_artigo_revogado"] = sum(r.artigo_revogado for r in topo)
        linha["ctx_artigo_alterado"] = sum(r.artigo_alterado for r in topo)
        linhas.append(linha)

    return pd.DataFrame(linhas)


def resumir(df: pd.DataFrame) -> pd.Series:
    colunas = ["mrr"] + [f"recall@{k}" for k in config.KS_AVALIADOS] + \
              [f"precisao@{k}" for k in config.KS_AVALIADOS]
    resumo = df[colunas].mean()
    resumo["perguntas"] = len(df)
    resumo["sem_nenhum_acerto"] = float((df["mrr"] == 0).mean())
    resumo["ctx_com_revogado"] = float((df["ctx_revogado"] > 0).mean())
    resumo["ctx_com_art_revogado"] = float((df["ctx_artigo_revogado"] > 0).mean())
    resumo["ctx_com_alterado"] = float((df["ctx_artigo_alterado"] > 0).mean())
    return resumo


def comparar_estrategias(indice, golden: list[dict], estrategias=("bm25", "denso", "hibrido")) -> pd.DataFrame:
    return pd.DataFrame(
        {e: resumir(avaliar_recuperacao(indice, golden, e)) for e in estrategias}
    ).T


# --------------------------------------------------------------------------
# Vigência — a parte que quase nenhum harness cobre
# --------------------------------------------------------------------------
def auditar_vigencia(resposta_citacoes: list[dict], chunks: pd.DataFrame) -> dict:
    """Confere cada citação da resposta contra o estado real da norma.

    Uma premissa foi testada e derrubada ao montar isto, e ela define os
    veredictos abaixo. Supunha-se que o perigo fosse o sistema citar
    redação superada. Não é: o BACEN publica texto **consolidado**, então
    o trecho recuperado já traz a redação vigente com marcador de
    atribuição. Citar artigo que sofreu alteração não é erro.

    Os erros de verdade, com contagem no corpus deste projeto:

        norma_revogada    a norma inteira foi revogada — 44,6% dos
                          normativos recuperáveis estão nesse estado
        artigo_revogado   a norma vale, mas o artigo citado foi revogado
                          por norma posterior e segue impresso no
                          documento. O mais traiçoeiro dos três.
        inexistente       a citação não corresponde a nada no corpus.
                          É invenção pura, e é o mais grave.

    `alterado` aparece como informação, não como falta.
    """
    indexado: dict[str, dict] = {}
    normas: set[str] = set()
    for _, r in chunks.iterrows():
        indexado.setdefault(_chave(r["rotulo_norma"], r["artigo"], r.get("parte", "")), r)
        normas.add(_normalizar(r["rotulo_norma"]))

    veredictos = []
    for c in resposta_citacoes:
        chave = _chave(c.get("norma", ""), c.get("artigo"), c.get("parte", ""))
        r = indexado.get(chave)

        if r is not None:
            if bool(r["revogado"]):
                v = "norma_revogada"
            elif bool(r.get("artigo_revogado", False)):
                v = "artigo_revogado"
            else:
                v = "vigente"
            alterado = bool(r["artigo_alterado"])
        elif _normalizar(c.get("norma", "")) in normas:
            v, alterado = "artigo_inexistente", False
        else:
            v, alterado = "inexistente", False

        veredictos.append({**c, "veredicto": v, "teve_nova_redacao": alterado})

    contagem = pd.Series([v["veredicto"] for v in veredictos]).value_counts().to_dict()
    total = max(len(veredictos), 1)
    inventadas = contagem.get("inexistente", 0) + contagem.get("artigo_inexistente", 0)

    return {
        "citacoes": veredictos,
        "n_citacoes": len(veredictos),
        "vigentes": contagem.get("vigente", 0),
        "norma_revogada": contagem.get("norma_revogada", 0),
        "artigo_revogado": contagem.get("artigo_revogado", 0),
        "inexistentes": inventadas,
        "com_nova_redacao": sum(v["teve_nova_redacao"] for v in veredictos),
        "taxa_vigente": contagem.get("vigente", 0) / total,
        # Confiável = citou alguma coisa, e tudo que citou existe e vale.
        "resposta_integra": bool(veredictos)
        and all(v["veredicto"] == "vigente" for v in veredictos),
    }


def fundamentacao(resposta: str, contexto: list[str], n_gram: int = 8) -> float:
    """Fração das sequências da resposta que aparecem no contexto.

    Heurística barata e sem LLM: se um trecho de 8 palavras da resposta
    não existe em nenhuma passagem recuperada, ou é reformulação ou é
    invenção. Não distingue as duas — por isso serve como triagem, não
    como veredicto, e o harness a usa junto com a auditoria de citação.
    """
    import unicodedata

    def normaliza(s):
        t = unicodedata.normalize("NFKD", s.lower())
        t = "".join(c for c in t if not unicodedata.combining(c))
        return t.split()

    palavras = normaliza(resposta)
    if len(palavras) < n_gram:
        return float("nan")

    fonte = " ".join(" ".join(normaliza(c)) for c in contexto)
    grams = [" ".join(palavras[i:i + n_gram]) for i in range(len(palavras) - n_gram + 1)]
    if not grams:
        return float("nan")
    return sum(g in fonte for g in grams) / len(grams)
