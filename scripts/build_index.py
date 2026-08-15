"""Constrói e guarda em disco os chunks e seus embeddings.

Separado do pipeline de avaliação porque é a etapa cara: roda uma vez e
o harness reusa. Em CPU, embutir o corpus leva alguns minutos.

Uso:
    python -m scripts.build_index [--estrategia artigo|janela]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import chunking, config, corpus, retrieval  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass


def caminhos(estrategia: str) -> tuple[Path, Path]:
    sufixo = "" if estrategia == "artigo" else f"_{estrategia}"
    return (
        config.DATA_PROCESSED / f"chunks{sufixo}.parquet",
        config.DATA_PROCESSED / f"embeddings{sufixo}.npy",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--estrategia", default="artigo", choices=["artigo", "janela"])
    args = p.parse_args()

    normativos = corpus.carregar()
    if not normativos:
        raise SystemExit("corpus vazio; rode a coleta antes (src.corpus.coletar)")

    r = corpus.resumo(normativos)
    print(f"corpus: {r['normativos']} normativos | {r['revogados']} revogados "
          f"({r['pct_revogados']:.1%}) | {r['chars_totais']:,} chars".replace(",", "."))

    df = chunking.construir(normativos, args.estrategia)
    print(f"chunks ({args.estrategia}): {len(df):,}".replace(",", "."))

    textos = chunking.texto_para_embedding(df)
    indice = retrieval.Indice(df)

    print(f"embutindo com {config.MODELO_EMBEDDING} ...")
    t0 = time.perf_counter()
    emb = indice.indexar(textos, verboso=True)
    print(f"  {emb.shape} em {time.perf_counter() - t0:.0f}s")

    p_chunks, p_emb = caminhos(args.estrategia)
    df.to_parquet(p_chunks)
    np.save(p_emb, emb)
    print(f"salvo: {p_chunks.name} e {p_emb.name}")


if __name__ == "__main__":
    main()
