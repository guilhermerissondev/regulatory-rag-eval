"""Testes da segmentação.

Se o chunk perder a etiqueta do artigo, a auditoria de vigência inteira
deixa de funcionar em silêncio — a resposta continua saindo, só que sem
como verificar. Estes testes travam essa propriedade.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import chunking  # noqa: E402
from src.corpus import Normativo, artigos_alterados, limpar_html  # noqa: E402


@pytest.fixture
def norma():
    texto = (
        "RESOLUÇÃO BCB Nº 99, DE 1 DE JANEIRO DE 2020\n\n"
        "Institui coisa nenhuma.\n\n"
        "CAPÍTULO I - DAS DISPOSIÇÕES GERAIS\n\n"
        "Art. 1º Esta resolução dispõe sobre o arranjo de pagamentos. "
        "Parágrafo único. O disposto no caput aplica-se a todos.\n\n"
        "Art. 2º O prazo para contestação é de 80 dias corridos, "
        "contados da data da transação que se pretende contestar.\n\n"
        "CAPÍTULO II - DO MECANISMO ESPECIAL\n\n"
        "Seção I - Da abertura\n\n"
        "Art. 3º-A A instituição deve notificar o usuário em até 24 horas.\n\n"
        "Art. 4º Revogado.\n"
    )
    return Normativo(
        id="99",
        tipo="Resolução BCB",
        numero="99",
        titulo="Resolução BCB Nº 99",
        assunto="Institui coisa nenhuma.",
        data="2020-01-01",
        revogado=False,
        cancelado=False,
        texto=texto,
        atualizacoes=[
            "Resolução BCB nº 120/2021 - Nova redação: art. 2º; art. 3º-A, caput.",
        ],
    )


# --------------------------------------------------------------------------
# Estrutura
# --------------------------------------------------------------------------
def test_separa_um_chunk_por_artigo(norma):
    chunks = chunking.por_artigo(norma)
    artigos = [c.artigo for c in chunks if c.artigo]
    assert artigos == ["1", "2", "3-A", "4"]


def test_preambulo_vira_chunk_sem_artigo(norma):
    chunks = chunking.por_artigo(norma)
    preambulo = [c for c in chunks if c.artigo is None]
    assert len(preambulo) == 1
    assert "Institui coisa nenhuma" in preambulo[0].texto


def test_artigo_carrega_paragrafo_unico_junto(norma):
    """Separar a regra da sua exceção é o erro que motiva o chunking por
    artigo. O parágrafo único tem que ficar com o caput."""
    c = next(c for c in chunking.por_artigo(norma) if c.artigo == "1")
    assert "Parágrafo único" in c.texto


def test_hierarquia_e_propagada(norma):
    chunks = {c.artigo: c for c in chunking.por_artigo(norma) if c.artigo}
    assert "CAPÍTULO I" in chunks["1"].capitulo
    assert "CAPÍTULO II" in chunks["3-A"].capitulo
    assert "Seção I" in chunks["3-A"].secao
    assert chunks["1"].secao == ""


def test_artigo_com_sufixo_e_reconhecido(norma):
    """Artigos como 3-A são inseridos por alteração posterior, então são
    justamente os que mais aparecem no histórico de mudanças."""
    assert any(c.artigo == "3-A" for c in chunking.por_artigo(norma))


# --------------------------------------------------------------------------
# Vigência
# --------------------------------------------------------------------------
def test_artigos_alterados_extraidos_do_historico(norma):
    assert artigos_alterados(norma) == {"2", "3-A"}


def test_marca_de_alteracao_chega_ao_chunk(norma):
    chunks = {c.artigo: c for c in chunking.por_artigo(norma) if c.artigo}
    assert chunks["2"].artigo_alterado is True
    assert chunks["3-A"].artigo_alterado is True
    assert chunks["1"].artigo_alterado is False


def test_janela_fixa_perde_a_rastreabilidade(norma):
    """O baseline não é só pior de recuperar — ele impossibilita a
    auditoria, porque nenhum chunk sabe de que artigo veio."""
    chunks = chunking.janela_fixa(norma, tamanho=200, overlap=20)
    assert len(chunks) > 1
    assert all(c.artigo is None for c in chunks)
    assert all(c.artigo_alterado is False for c in chunks)


def test_citacao_montada_a_partir_do_chunk(norma):
    c = next(c for c in chunking.por_artigo(norma) if c.artigo == "2")
    assert c.citacao == "Resolução BCB nº 99, art. 2"


# --------------------------------------------------------------------------
# Divisão de artigo longo
# --------------------------------------------------------------------------
def test_artigo_longo_e_dividido_mantendo_o_rotulo():
    longo = "Art. 7º " + ("Texto de teste com muitas palavras. " * 300)
    n = Normativo(
        id="1", tipo="Resolução BCB", numero="1", titulo="t", assunto="a",
        data="2020-01-01", revogado=False, cancelado=False, texto=longo,
    )
    chunks = chunking.por_artigo(n)
    assert len(chunks) > 1
    assert all(c.artigo == "7" for c in chunks)
    assert all(len(c.texto) <= 2200 for c in chunks)


def test_artigo_curto_sobrevive_ao_piso():
    """"Art. 4º Revogado." tem 17 caracteres e precisa ser recuperável —
    é a resposta certa para "o art. 4 ainda vale?"."""
    n = Normativo(
        id="1", tipo="Resolução BCB", numero="1", titulo="t", assunto="a",
        data="2020-01-01", revogado=False, cancelado=False,
        texto="Art. 4º Revogado.\n\nArt. 5º " + ("palavra " * 40),
    )
    artigos = [c.artigo for c in chunking.por_artigo(n)]
    assert artigos == ["4", "5"]


def test_fragmento_solto_sem_artigo_e_descartado():
    n = Normativo(
        id="1", tipo="Resolução BCB", numero="1", titulo="t", assunto="a",
        data="2020-01-01", revogado=False, cancelado=False,
        texto="Seção II\n\nArt. 1º " + ("palavra " * 40),
    )
    chunks = chunking.por_artigo(n)
    assert [c.artigo for c in chunks] == ["1"]  # o preâmbulo curto sumiu


# --------------------------------------------------------------------------
# Limpeza de HTML
# --------------------------------------------------------------------------
def test_limpar_html_desfaz_escape_duplo():
    assert limpar_html("Art. 1&amp;#186; teste") == "Art. 1º teste"


def test_limpar_html_remove_tags_e_normaliza_espaco():
    bruto = '<div class="ExternalClass123"><p>Art.  1º</p><br/><p>Texto</p></div>'
    assert limpar_html(bruto) == "Art. 1º\n\nTexto"


def test_limpar_html_aceita_vazio():
    assert limpar_html(None) == ""
    assert limpar_html("") == ""


# --------------------------------------------------------------------------
# Texto para embedding
# --------------------------------------------------------------------------
def test_prefixo_e_cabecalho_no_texto_embutido(norma):
    from src import config

    df = chunking.construir([norma], "artigo")
    textos = chunking.texto_para_embedding(df)
    assert all(t.startswith(config.PREFIXO_PASSAGEM) for t in textos)
    assert any("art. 3-A" in t and "CAPÍTULO II" in t for t in textos)
