"""Testes da resolução de versões empilhadas.

Se este módulo errar, o sistema entrega texto revogado com aparência de
texto vigente — a falha mais cara possível em domínio regulado, e a mais
difícil de perceber, porque a citação confere com o documento oficial.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import versioning  # noqa: E402

# Reprodução fiel do art. 39 do Regulamento do Pix, que empilha três
# gerações do inciso II.
ART39 = (
    "Art. 39. Uma transação no âmbito do Pix deverá ser rejeitada quando: "
    "I - houver fundada suspeita de fraude; ou "
    "I - houver fundada suspeita de fraude; (Redação dada, a partir de 1º/2/2022, "
    "pela Resolução BCB nº 181, de 25/1/2022.) "
    "II - houver problemas na identificação do usuário recebedor. "
    "II - houver problemas na identificação do usuário recebedor; ou (Redação dada, "
    "a partir de 1º/2/2022, pela Resolução BCB nº 181, de 25/1/2022.) "
    "II - houver problemas na identificação do usuário recebedor; (Redação dada pela "
    "Resolução BCB nº 402, de 22/7/2024.) "
    "III - o agente de saque não tiver sido habilitado."
)


# --------------------------------------------------------------------------
# Detecção
# --------------------------------------------------------------------------
def test_detecta_empilhamento():
    assert versioning.tem_empilhamento(ART39) is True


def test_artigo_sem_alteracao_nao_e_marcado():
    limpo = (
        "Art. 8º O DICT entrará em funcionamento: "
        "I - no dia 5 de outubro de 2020, em operação restrita; e "
        "II - no dia 16 de novembro de 2020, em operação plena."
    )
    assert versioning.tem_empilhamento(limpo) is False


def test_rotulo_repetido_sem_atribuicao_nao_conta():
    """Lista aninhada legítima pode repetir rótulo. Sem marcador de
    redação, não é empilhamento de versão."""
    texto = "Art. 1º Aplica-se: I - ao caso A; II - ao caso B. Parágrafo único. Nos termos: I - do anexo."
    assert versioning.tem_empilhamento(texto) is False


# --------------------------------------------------------------------------
# Resolução
# --------------------------------------------------------------------------
def test_mantem_apenas_a_ultima_versao():
    r = versioning.resolver(ART39)
    assert r.count("houver problemas na identificação") == 1
    assert r.count("houver fundada suspeita de fraude") == 1


def test_versao_mantida_e_a_mais_recente():
    r = versioning.resolver(ART39)
    assert "Resolução BCB nº 402" in r       # a última alteração do inciso II
    assert "; ou (Redação dada, a partir de 1º/2/2022" not in r


def test_caput_preservado():
    assert versioning.resolver(ART39).startswith("Art. 39.")


def test_dispositivo_sem_versao_alternativa_passa_intacto():
    assert "III - o agente de saque não tiver sido habilitado." in versioning.resolver(ART39)


def test_ordem_dos_dispositivos_preservada():
    r = versioning.resolver(ART39)
    assert r.index("I - houver") < r.index("II - houver") < r.index("III - o agente")


def test_texto_sem_dispositivo_passa_intacto():
    t = "Art. 2º Fica aprovado o Regulamento anexo."
    assert versioning.resolver(t) == t


# --------------------------------------------------------------------------
# Revogação
# --------------------------------------------------------------------------
def test_dispositivo_revogado_e_removido():
    texto = (
        "Art. 5º Aplica-se o seguinte: "
        "I - a regra antiga; "
        "I - (Revogado, a partir de 1º/3/2023, pela Resolução BCB nº 300.) "
        "II - a regra que continua valendo."
    )
    r = versioning.resolver(texto)
    assert "a regra antiga" not in r
    assert "a regra que continua valendo" in r


def test_revogado_pode_ser_mantido_sob_demanda():
    texto = (
        "Art. 5º Aplica-se: I - a regra antiga; "
        "I - (Revogado, a partir de 1º/3/2023, pela Resolução BCB nº 300.)"
    )
    r = versioning.resolver(texto, manter_revogados=True)
    assert "Revogado" in r


# --------------------------------------------------------------------------
# Limpeza de atribuições
# --------------------------------------------------------------------------
def test_limpar_atribuicoes_remove_parenteticos():
    r = versioning.limpar_atribuicoes(versioning.resolver(ART39))
    assert "Redação dada" not in r
    assert "houver fundada suspeita de fraude" in r


def test_limpeza_depois_da_resolucao_nao_perde_dispositivo():
    r = versioning.limpar_atribuicoes(versioning.resolver(ART39))
    for rot in ("I - ", "II - ", "III - "):
        assert rot in r


# --------------------------------------------------------------------------
# Estatísticas
# --------------------------------------------------------------------------
def test_estatisticas_reportam_reducao():
    e = versioning.estatisticas([ART39, "Art. 1º Texto simples sem dispositivo."])
    assert e["chunks"] == 2
    assert e["com_empilhamento"] == 1
    assert e["reducao"] > 0
    assert e["chars_depois"] < e["chars_antes"]
