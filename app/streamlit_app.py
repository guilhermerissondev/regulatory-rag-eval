"""Assistente de conformidade, com a auditoria à vista.

O ponto do app não é o chat. É mostrar, ao lado de cada resposta, o
estado de vigência de tudo que foi recuperado e citado — porque num
corpus onde 44,6% das normas estão revogadas, uma resposta fluente sem
essa checagem é um passivo, não um produto.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, metrics, retrieval, versioning  # noqa: E402

st.set_page_config(page_title="RAG Regulatório", page_icon="§", layout="wide")

VERDE, VERMELHO, AMBAR, CINZA = "#27ae60", "#c0392b", "#d68910", "#7f8c8d"


@st.cache_resource
def carregar():
    if not config.CHUNKS_PARQUET.exists():
        return None, None
    df = pd.read_parquet(config.CHUNKS_PARQUET)
    emb = np.load(config.EMBEDDINGS_NPY) if config.EMBEDDINGS_NPY.exists() else None
    return retrieval.Indice(df, embeddings=emb), df


indice, chunks = carregar()
if indice is None:
    st.error("Índice não encontrado. Rode antes:\n\n```\npython -m scripts.build_index\n```")
    st.stop()


def selo(texto: str, cor: str) -> str:
    return (
        f"<span style='background:{cor};color:white;padding:1px 7px;"
        f"border-radius:9px;font-size:0.72rem;white-space:nowrap'>{texto}</span>"
    )


def selos_do_trecho(r) -> str:
    s = []
    if r.revogado:
        s.append(selo("NORMA REVOGADA", VERMELHO))
    elif r.artigo_revogado:
        s.append(selo("ARTIGO REVOGADO", VERMELHO))
    else:
        s.append(selo("vigente", VERDE))
    if versioning.tem_empilhamento(r.texto):
        s.append(selo("versões empilhadas", AMBAR))
    elif r.artigo_alterado:
        s.append(selo("teve nova redação", CINZA))
    return " ".join(s)


# --------------------------------------------------------------------------
st.sidebar.header("Recuperação")
estrategia = st.sidebar.radio(
    "Estratégia", ["hibrido", "denso", "bm25"],
    format_func=lambda s: {"hibrido": "Híbrida (RRF)", "denso": "Densa (E5)",
                           "bm25": "Léxica (BM25)"}[s],
)
k = st.sidebar.slider("Trechos recuperados", 3, 15, config.TOP_K)
resolver = st.sidebar.checkbox(
    "Resolver versões empilhadas", value=True,
    help="Mantém apenas a redação vigente de cada dispositivo antes de "
         "montar o contexto do modelo.",
)
gerar = st.sidebar.checkbox("Gerar resposta com LLM", value=False,
                            help="Requer GEMINI_API_KEY no ambiente.")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"{len(chunks):,} trechos · {chunks['revogado'].mean():.1%} de norma revogada · "
    f"{chunks['artigo_revogado'].mean():.1%} de artigo revogado".replace(",", ".")
)

st.title("Assistente de conformidade — Pix")
st.caption(
    "Corpus de normativos do BACEN. Cada trecho vem com o estado de vigência, "
    "porque num corpus assim a resposta certa e a resposta perigosa se parecem."
)

exemplos = [
    "Com quanta antecedência deve ser comunicado o encerramento da prestação de serviço?",
    "Em que situações uma transação Pix deve ser rejeitada?",
    "A partir de quantas contas ativas a participação no Pix é obrigatória?",
    "Qual a alíquota de imposto de renda sobre Pix acima de mil reais?",
]
col1, col2 = st.columns([3, 1])
with col2:
    escolha = st.selectbox("Exemplos", ["—"] + exemplos, label_visibility="collapsed")
with col1:
    pergunta = st.text_input(
        "Pergunta", value="" if escolha == "—" else escolha,
        placeholder="Ex.: qual o prazo para comunicar encerramento?",
        label_visibility="collapsed",
    )

if not pergunta:
    st.info(
        "A última pergunta da lista é uma armadilha: não existe resposta no corpus. "
        "Um sistema honesto recusa; um sistema fluente inventa um artigo."
    )
    st.stop()

resultados = indice.buscar(pergunta, estrategia=estrategia, k=k)

# --------------------------------------------------------------------------
if gerar:
    st.subheader("Resposta")
    try:
        from src.generation import Gerador

        with st.spinner("consultando o modelo..."):
            r = Gerador().responder(pergunta, resultados, resolver_versoes=resolver)
    except RuntimeError as e:
        st.error(str(e))
        r = None

    if r and r.erro:
        st.error(f"falha na chamada: {r.erro[:220]}")
    elif r:
        st.write(r.texto or "_(sem resposta)_")

        aud = metrics.auditar_vigencia(r.citacoes, chunks)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Citações", aud["n_citacoes"])
        c2.metric("Vigentes", aud["vigentes"])
        c3.metric("Revogadas", aud["norma_revogada"] + aud["artigo_revogado"])
        c4.metric("Inexistentes", aud["inexistentes"])

        if aud["n_citacoes"] == 0:
            st.warning("O modelo não citou nenhuma norma — para pergunta sem base no corpus, é o comportamento correto.")
        elif aud["resposta_integra"]:
            st.success("Todas as citações existem no corpus e estão vigentes.")
        else:
            st.error("Há citação revogada ou inexistente. Detalhe abaixo.")

        for c in aud["citacoes"]:
            rot = {"vigente": (VERDE, "vigente"),
                   "norma_revogada": (VERMELHO, "norma revogada"),
                   "artigo_revogado": (VERMELHO, "artigo revogado"),
                   "inexistente": (VERMELHO, "não existe no corpus"),
                   "artigo_inexistente": (VERMELHO, "artigo não encontrado")}[c["veredicto"]]
            art = f", art. {c.get('artigo')}" if c.get("artigo") else ""
            st.markdown(
                f"- {c.get('norma','?')}{art} &nbsp; {selo(rot[1], rot[0])}",
                unsafe_allow_html=True,
            )

        st.caption(
            f"{r.tokens_entrada} tokens de entrada · {r.tokens_saida} de saída · "
            f"US$ {r.custo_estimado_usd:.5f} · {r.latencia_s:.1f}s"
        )
    st.markdown("---")

# --------------------------------------------------------------------------
st.subheader(f"Trechos recuperados ({estrategia})")

n_revogados = sum(r.revogado or r.artigo_revogado for r in resultados)
n_empilhados = sum(versioning.tem_empilhamento(r.texto) for r in resultados)
if n_revogados:
    st.warning(
        f"{n_revogados} de {len(resultados)} trechos recuperados vêm de norma ou "
        f"artigo revogado. Sem a checagem, isso entraria no contexto do modelo "
        f"indistinguível do resto."
    )
if n_empilhados:
    st.info(
        f"{n_empilhados} trecho(s) trazem várias gerações do mesmo dispositivo "
        f"coladas. {'A resolução está ligada.' if resolver else 'A resolução está DESLIGADA — o modelo veria texto revogado.'}"
    )

for r in resultados:
    with st.expander(f"{r.posicao}. {r.citacao}", expanded=(r.posicao <= 2)):
        st.markdown(selos_do_trecho(r), unsafe_allow_html=True)
        empilhado = versioning.tem_empilhamento(r.texto)
        if empilhado and resolver:
            aba1, aba2 = st.tabs(["Redação vigente", "Texto original (empilhado)"])
            aba1.write(versioning.resolver(r.texto))
            aba2.write(r.texto)
        else:
            st.write(versioning.resolver(r.texto) if resolver else r.texto)
        st.caption(f"score {r.score:.4f}")
