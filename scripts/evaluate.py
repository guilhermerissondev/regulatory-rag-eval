"""Harness de avaliação.

Uso:
    python -m scripts.evaluate            # recuperação + geração
    python -m scripts.evaluate --so-recuperacao   # sem gastar token

Três blocos, na ordem em que uma falha importa:

    1. recuperação   o trecho certo apareceu? (não usa LLM)
    2. contaminação  o contexto entregue tem norma revogada dentro?
    3. geração       a resposta cita o que existe e ainda vale?
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, metrics, retrieval, versioning  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def log(m: str = "") -> None:
    print(f"  {m}" if m else "", flush=True)


def carregar_indice() -> tuple[retrieval.Indice, pd.DataFrame]:
    df = pd.read_parquet(config.CHUNKS_PARQUET)
    emb = np.load(config.EMBEDDINGS_NPY)
    return retrieval.Indice(df, embeddings=emb), df


def carregar_golden() -> list[dict]:
    return json.loads(config.GOLDEN_JSON.read_text(encoding="utf-8"))["perguntas"]


# --------------------------------------------------------------------------
def bloco_recuperacao(indice, golden, chunks) -> pd.DataFrame:
    print("\n=== 1. Recuperação ===")
    # Perguntas sem resposta no corpus não entram: não existe trecho certo
    # a recuperar, e incluí-las contaminaria o recall com NaN.
    respondiveis = [g for g in golden if g.get("artigos_relevantes")]
    log(f"{len(respondiveis)} perguntas com âncora | "
        f"{len(golden) - len(respondiveis)} sem resposta no corpus (avaliadas na geração)")

    comp = metrics.comparar_estrategias(indice, respondiveis)
    cols = ["mrr", "recall@1", "recall@3", "recall@5", "recall@8",
            "sem_nenhum_acerto", "ctx_com_revogado"]
    print()
    print(comp[cols].to_string(float_format=lambda v: f"{v:.3f}"))
    comp.to_csv(config.REPORTS / "recuperacao_por_estrategia.csv")

    melhor = comp["mrr"].idxmax()
    log("")
    log(f"melhor estratégia por MRR: {melhor} ({comp.loc[melhor, 'mrr']:.3f})")

    detalhe = metrics.avaliar_recuperacao(indice, respondiveis, melhor)
    detalhe.to_csv(config.REPORTS / "recuperacao_por_pergunta.csv", index=False)

    piores = detalhe[detalhe["mrr"] == 0]
    if len(piores):
        log(f"perguntas sem nenhum acerto em {max(config.KS_AVALIADOS)}: "
            + ", ".join(piores["id"]))
    return comp


def bloco_contaminacao(indice, golden, chunks) -> dict:
    print("\n=== 2. Contaminação do contexto ===")
    respondiveis = [g for g in golden if g.get("artigos_relevantes")]
    det = metrics.avaliar_recuperacao(indice, respondiveis, "hibrido")

    empilhados = 0
    for g in respondiveis:
        for r in indice.buscar(g["pergunta"], "hibrido", k=config.TOP_K):
            if versioning.tem_empilhamento(r.texto):
                empilhados += 1

    total_trechos = len(respondiveis) * config.TOP_K
    res = {
        "perguntas": len(respondiveis),
        "trechos_por_pergunta": config.TOP_K,
        "pct_perguntas_com_revogado_no_ctx": float((det["ctx_revogado"] > 0).mean()),
        "pct_perguntas_com_art_revogado_no_ctx": float((det["ctx_artigo_revogado"] > 0).mean()),
        "trechos_empilhados": empilhados,
        "pct_trechos_empilhados": empilhados / max(total_trechos, 1),
        "corpus_pct_revogado": float(chunks["revogado"].mean()),
    }
    log(f"chunks do corpus vindos de norma revogada : {res['corpus_pct_revogado']:.1%}")
    log(f"perguntas cujo contexto contém norma revogada: "
        f"{res['pct_perguntas_com_revogado_no_ctx']:.1%}")
    log(f"perguntas cujo contexto contém artigo revogado: "
        f"{res['pct_perguntas_com_art_revogado_no_ctx']:.1%}")
    log(f"trechos com versões empilhadas: {empilhados}/{total_trechos} "
        f"({res['pct_trechos_empilhados']:.1%})")
    return res


def bloco_geracao(indice, golden, chunks) -> pd.DataFrame:
    print("\n=== 3. Geração ===")
    from src.generation import Gerador

    try:
        ger = Gerador()
    except RuntimeError as e:
        log(f"pulando: {e}")
        return pd.DataFrame()

    modos = [
        ("rag_resolvido", dict(rag=True, resolver=True)),
        ("rag_bruto", dict(rag=True, resolver=False)),
        ("sem_contexto", dict(rag=False, resolver=False)),
    ]

    total = len(modos) * len(golden)
    log(f"{total} chamadas a {config.RPM_LIMITE} req/min -> ~{total * 60 / config.RPM_LIMITE / 60:.0f} min")

    linhas = []
    for nome, cfg in modos:
        log(f"rodando {nome} ...")
        for g in golden:
            if cfg["rag"]:
                res = indice.buscar(g["pergunta"], "hibrido", k=config.TOP_K)
                r = ger.responder(g["pergunta"], res, resolver_versoes=cfg["resolver"])
            else:
                r = ger.responder_sem_contexto(g["pergunta"])

            aud = metrics.auditar_vigencia(r.citacoes, chunks)
            fund = metrics.fundamentacao(r.texto, r.contexto_usado) if r.contexto_usado else float("nan")

            linhas.append({
                "modo": nome,
                "id": g["id"],
                "armadilha": g.get("armadilha") or "",
                "tem_resposta_no_corpus": bool(g.get("artigos_relevantes")),
                "declarou_suficiente": r.suficiente,
                "n_citacoes": aud["n_citacoes"],
                "vigentes": aud["vigentes"],
                "norma_revogada": aud["norma_revogada"],
                "artigo_revogado": aud["artigo_revogado"],
                "inexistentes": aud["inexistentes"],
                "resposta_integra": aud["resposta_integra"],
                "fundamentacao": fund,
                "tokens_entrada": r.tokens_entrada,
                "tokens_saida": r.tokens_saida,
                "latencia_s": r.latencia_s,
                "custo_usd": r.custo_estimado_usd,
                "erro": r.erro,
                "texto": r.texto[:400],
                # Citações cruas ficam salvas para que reauditar não exija
                # gastar token de novo — o que importou quando um bug na
                # normalização obrigou a refazer a auditoria inteira.
                "citacoes_json": json.dumps(r.citacoes, ensure_ascii=False),
            })

    df = pd.DataFrame(linhas)
    df.to_csv(config.REPORTS / "geracao_por_pergunta.csv", index=False)

    com_resp = df[df["tem_resposta_no_corpus"]]
    sem_resp = df[~df["tem_resposta_no_corpus"]]

    resumo = pd.DataFrame({
        "citacoes_por_resposta": com_resp.groupby("modo")["n_citacoes"].mean(),
        "pct_integra": com_resp.groupby("modo")["resposta_integra"].mean(),
        "citacoes_inexistentes": com_resp.groupby("modo")["inexistentes"].sum(),
        "citacoes_norma_revogada": com_resp.groupby("modo")["norma_revogada"].sum(),
        "citacoes_art_revogado": com_resp.groupby("modo")["artigo_revogado"].sum(),
        "fundamentacao": com_resp.groupby("modo")["fundamentacao"].mean(),
        "custo_usd_medio": df.groupby("modo")["custo_usd"].mean(),
        "latencia_s": df.groupby("modo")["latencia_s"].mean(),
    })
    print()
    print(resumo.to_string(float_format=lambda v: f"{v:.4f}"))
    resumo.to_csv(config.REPORTS / "geracao_resumo.csv")

    print()
    log("perguntas SEM resposta no corpus (o certo é recusar):")
    for modo, sub in sem_resp.groupby("modo"):
        recusou = (~sub["declarou_suficiente"] | (sub["n_citacoes"] == 0)).mean()
        inventou = sub["inexistentes"].sum()
        log(f"  {modo:<15} recusou {recusou:>5.0%} | citações inventadas: {inventou}")

    erros = df[df["erro"] != ""]
    if len(erros):
        log(f"chamadas com erro: {len(erros)}")
        log(f"  exemplo: {erros.iloc[0]['erro'][:150]}")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--so-recuperacao", action="store_true")
    args = p.parse_args()

    indice, chunks = carregar_indice()
    golden = carregar_golden()
    print(f"corpus: {len(chunks):,} chunks | golden: {len(golden)} perguntas".replace(",", "."))

    comp = bloco_recuperacao(indice, golden, chunks)
    cont = bloco_contaminacao(indice, golden, chunks)

    ger = pd.DataFrame()
    if not args.so_recuperacao:
        ger = bloco_geracao(indice, golden, chunks)

    saida = {
        "corpus": {
            "chunks": int(len(chunks)),
            "pct_chunks_norma_revogada": float(chunks["revogado"].mean()),
            "pct_chunks_artigo_revogado": float(chunks["artigo_revogado"].mean()),
        },
        "recuperacao": comp.to_dict("index"),
        "contaminacao": cont,
    }
    if len(ger):
        com_resp = ger[ger["tem_resposta_no_corpus"]]
        saida["geracao"] = {
            m: {
                "pct_integra": float(s["resposta_integra"].mean()),
                "citacoes_inexistentes": int(s["inexistentes"].sum()),
                "citacoes_norma_revogada": int(s["norma_revogada"].sum()),
                "citacoes_art_revogado": int(s["artigo_revogado"].sum()),
                "custo_usd_medio": float(ger[ger["modo"] == m]["custo_usd"].mean()),
            }
            for m, s in com_resp.groupby("modo")
        }

    (config.REPORTS / "metricas.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(f"\nrelatórios em {config.REPORTS}")


if __name__ == "__main__":
    main()
