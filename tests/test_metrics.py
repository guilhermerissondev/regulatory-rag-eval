"""Testes das métricas de recuperação e da auditoria de vigência.

A auditoria é o que separa este projeto de um chatbot: ela decide se uma
resposta bonita é confiável ou é passivo jurídico. Se ela errar, erra
silenciosamente.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import metrics  # noqa: E402


@pytest.fixture
def corpus():
    """Quatro dispositivos cobrindo os estados possíveis."""
    return pd.DataFrame([
        {"rotulo_norma": "Resolução BCB nº 1", "parte": "", "artigo": "3",
         "revogado": False, "artigo_revogado": False, "artigo_alterado": False},
        {"rotulo_norma": "Resolução BCB nº 1", "parte": "Regulamento anexo", "artigo": "3",
         "revogado": False, "artigo_revogado": False, "artigo_alterado": True},
        {"rotulo_norma": "Resolução BCB nº 1", "parte": "Regulamento anexo", "artigo": "50",
         "revogado": False, "artigo_revogado": True, "artigo_alterado": False},
        {"rotulo_norma": "Instrução Normativa BCB nº 184", "parte": "", "artigo": "1",
         "revogado": True, "artigo_revogado": False, "artigo_alterado": False},
    ])


# --------------------------------------------------------------------------
# Normalização e identidade
# --------------------------------------------------------------------------
def test_normalizacao_iguala_variantes_de_numero():
    assert metrics._normalizar("Resolução BCB N° 1") == metrics._normalizar("Resolução BCB nº 1")


def test_parte_desambigua_artigos_homonimos():
    """O caso que motivou o campo: dois 'art. 3' no mesmo documento."""
    corpo = metrics._chave("Resolução BCB nº 1", "3", "")
    anexo = metrics._chave("Resolução BCB nº 1", "3", "Regulamento anexo")
    assert corpo != anexo


# --------------------------------------------------------------------------
# Auditoria de vigência
# --------------------------------------------------------------------------
def test_citacao_vigente(corpus):
    r = metrics.auditar_vigencia([{"norma": "Resolução BCB nº 1", "artigo": "3"}], corpus)
    assert r["vigentes"] == 1
    assert r["resposta_integra"] is True


def test_citacao_de_norma_revogada(corpus):
    r = metrics.auditar_vigencia(
        [{"norma": "Instrução Normativa BCB nº 184", "artigo": "1"}], corpus
    )
    assert r["norma_revogada"] == 1
    assert r["resposta_integra"] is False


def test_citacao_de_artigo_revogado_dentro_de_norma_viva(corpus):
    """O caso mais traiçoeiro: a norma vale, o artigo não."""
    r = metrics.auditar_vigencia(
        [{"norma": "Resolução BCB nº 1", "artigo": "50", "parte": "Regulamento anexo"}],
        corpus,
    )
    assert r["artigo_revogado"] == 1
    assert r["resposta_integra"] is False


def test_citacao_inventada(corpus):
    r = metrics.auditar_vigencia([{"norma": "Resolução BCB nº 999", "artigo": "7"}], corpus)
    assert r["inexistentes"] == 1
    assert r["resposta_integra"] is False


def test_artigo_inexistente_em_norma_existente(corpus):
    r = metrics.auditar_vigencia([{"norma": "Resolução BCB nº 1", "artigo": "888"}], corpus)
    assert r["inexistentes"] == 1


def test_artigo_alterado_nao_e_erro(corpus):
    """Premissa corrigida no meio do projeto: o BACEN publica texto
    consolidado, então citar dispositivo que teve nova redação está
    certo. Isso é informação, não falta."""
    r = metrics.auditar_vigencia(
        [{"norma": "Resolução BCB nº 1", "artigo": "3", "parte": "Regulamento anexo"}],
        corpus,
    )
    assert r["vigentes"] == 1
    assert r["com_nova_redacao"] == 1
    assert r["resposta_integra"] is True


def test_resposta_sem_citacao_nao_e_integra(corpus):
    """Não citar nada pode ser a resposta certa para pergunta sem base,
    mas não conta como resposta íntegra — não há o que auditar."""
    r = metrics.auditar_vigencia([], corpus)
    assert r["n_citacoes"] == 0
    assert r["resposta_integra"] is False


def test_mistura_de_veredictos(corpus):
    r = metrics.auditar_vigencia([
        {"norma": "Resolução BCB nº 1", "artigo": "3"},
        {"norma": "Instrução Normativa BCB nº 184", "artigo": "1"},
        {"norma": "Resolução BCB nº 999", "artigo": "7"},
    ], corpus)
    assert (r["vigentes"], r["norma_revogada"], r["inexistentes"]) == (1, 1, 1)
    assert r["taxa_vigente"] == pytest.approx(1 / 3)
    assert r["resposta_integra"] is False


# --------------------------------------------------------------------------
# Métricas de ranking
# --------------------------------------------------------------------------
def test_rr_posicao_do_primeiro_acerto():
    assert metrics.rr([False, False, True, True]) == pytest.approx(1 / 3)
    assert metrics.rr([True]) == 1.0
    assert metrics.rr([False, False]) == 0.0


def test_recall_usa_total_exigido_como_denominador():
    """Se a pergunta exige 2 artigos e só 1 foi recuperado, recall é 0,5
    — mesmo que o segundo nem exista no índice."""
    assert metrics.recall_em_k([True, False, False], total_relevantes=2, k=3) == pytest.approx(0.5)


def test_recall_cresce_com_k():
    marcas = [False, False, True]
    assert metrics.recall_em_k(marcas, 1, 1) == 0.0
    assert metrics.recall_em_k(marcas, 1, 3) == 1.0


def test_precisao_em_k():
    assert metrics.precisao_em_k([True, False, True, False], 4) == pytest.approx(0.5)
    assert metrics.precisao_em_k([], 0) == 0.0


# --------------------------------------------------------------------------
# Fundamentação
# --------------------------------------------------------------------------
def test_fundamentacao_alta_quando_copia_o_contexto():
    ctx = ["O prazo para comunicar o encerramento da prestação de serviço é de noventa dias corridos."]
    resp = "O prazo para comunicar o encerramento da prestação de serviço é de noventa dias corridos."
    assert metrics.fundamentacao(resp, ctx) == pytest.approx(1.0)


def test_fundamentacao_baixa_quando_inventa():
    ctx = ["O prazo para comunicar o encerramento da prestação de serviço é de noventa dias."]
    resp = "A multa aplicável ao participante infrator corresponde a dois por cento do valor total da operação."
    assert metrics.fundamentacao(resp, ctx) < 0.2


def test_fundamentacao_indefinida_para_resposta_curta():
    import math

    assert math.isnan(metrics.fundamentacao("Sim.", ["contexto qualquer"]))
