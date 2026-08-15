"""Geração com citação obrigatória.

Duas decisões de projeto moldam este módulo.

**Saída estruturada em vez de texto livre.** Um parágrafo bonito com
"conforme a Resolução BCB nº 1" no meio é impossível de auditar
programaticamente. Exigindo uma lista de citações com norma e artigo em
campos separados, cada afirmação vira algo que o harness cruza com o
corpus. A auditoria de vigência só existe por causa disso.

**Baseline sem contexto.** O modelo já viu regulação do Pix no
pré-treino e responde sem RAG nenhum — com fluência igual e número de
artigo inventado. Comparar as duas respostas é a única forma de mostrar
o que a recuperação de fato acrescenta, em vez de assumir.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from . import config

INSTRUCAO = """Você é um assistente de conformidade regulatória do Sistema Financeiro Nacional.

Responda à pergunta usando EXCLUSIVAMENTE os trechos normativos fornecidos.

Regras inegociáveis:
1. Cada afirmação da resposta precisa vir de um trecho fornecido.
2. Cite norma e artigo exatamente como aparecem no trecho. Não invente número de artigo.
3. Se os trechos não permitirem responder, diga isso explicitamente e devolve a lista de citações vazia. Responder "não sei" é resultado correto.
4. Não recorra a conhecimento próprio sobre a regulação. Se não está nos trechos, não existe.

Responda em JSON válido, sem cercas de código, no formato:
{"resposta": "...", "citacoes": [{"norma": "Resolução BCB nº 1", "artigo": "97"}], "suficiente": true}

O campo "suficiente" indica se os trechos bastavam para responder."""

INSTRUCAO_SEM_CONTEXTO = """Você é um assistente de conformidade regulatória do Sistema Financeiro Nacional.

Responda à pergunta com base no seu conhecimento. Cite as normas e artigos aplicáveis.

Responda em JSON válido, sem cercas de código, no formato:
{"resposta": "...", "citacoes": [{"norma": "Resolução BCB nº 1", "artigo": "97"}], "suficiente": true}"""


@dataclass
class Resposta:
    pergunta: str
    texto: str
    citacoes: list[dict] = field(default_factory=list)
    suficiente: bool = True
    contexto_usado: list[str] = field(default_factory=list)
    tokens_entrada: int = 0
    tokens_saida: int = 0
    latencia_s: float = 0.0
    modo: str = "rag"
    versoes_resolvidas: bool = True
    erro: str = ""

    @property
    def custo_estimado_usd(self) -> float:
        """Preço de tabela do Gemini 2.5 Flash em ago/2026, por milhão de
        tokens. Está aqui porque custo por pergunta é decisão de produto:
        um harness que ignora isso não sabe dizer se a solução escala."""
        return (self.tokens_entrada * 0.30 + self.tokens_saida * 2.50) / 1_000_000


def _extrair_json(bruto: str) -> dict:
    """O modelo às vezes embrulha o JSON em cerca de código, mesmo quando
    instruído a não fazer. Tolerar isso é mais barato que reprocessar."""
    t = (bruto or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.M).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"resposta": t, "citacoes": [], "suficiente": False}


class LimitadorDeTaxa:
    """Espaça as chamadas para caber no limite do provedor.

    A camada gratuita do Gemini permite 5 requisições por minuto por
    modelo. Sem limitador, uma bateria de avaliação queima a cota nos
    primeiros segundos e o resto das chamadas volta vazia — foi o que
    aconteceu na primeira execução deste harness: 32 de 54 chamadas
    falharam com 429 e as métricas saíram sem significado.

    Métrica calculada sobre resposta que nunca chegou é pior que métrica
    nenhuma, porque parece resultado.
    """

    def __init__(self, por_minuto: int = None):
        self.intervalo = 60.0 / max(por_minuto or config.RPM_LIMITE, 1)
        self._ultima = 0.0

    def aguardar(self) -> None:
        espera = self.intervalo - (time.perf_counter() - self._ultima)
        if espera > 0:
            time.sleep(espera)
        self._ultima = time.perf_counter()


def _segundos_de_espera(erro: str) -> float:
    """Extrai o retryDelay que o próprio 429 informa."""
    m = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", erro)
    if m:
        return float(m.group(1)) + 1.0
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", erro)
    return float(m.group(1)) + 1.0 if m else 30.0


class Gerador:
    def __init__(self, modelo: str = None, api_key: str = None):
        from google import genai

        chave = api_key or os.environ.get(config.VARIAVEL_CHAVE)
        if not chave:
            raise RuntimeError(
                f"variável {config.VARIAVEL_CHAVE} não encontrada no ambiente.\n"
                "  Defina com:  setx GEMINI_API_KEY \"...\"  e reabra o terminal."
            )
        self.modelo = modelo or config.MODELO_LLM
        self._cliente = genai.Client(api_key=chave)
        self._limitador = LimitadorDeTaxa()

    def _chamar(self, instrucao: str, conteudo: str) -> tuple[str, int, int, float, str]:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=instrucao,
            temperature=config.TEMPERATURA,
            max_output_tokens=config.MAX_TOKENS_RESPOSTA,
            response_mime_type="application/json",
        )

        ultimo_erro = ""
        for tentativa in range(config.MAX_TENTATIVAS):
            self._limitador.aguardar()
            t0 = time.perf_counter()
            try:
                r = self._cliente.models.generate_content(
                    model=self.modelo, contents=conteudo, config=cfg
                )
            except Exception as e:  # noqa: BLE001
                ultimo_erro = f"{type(e).__name__}: {e}"

                # Cota estourada: o próprio erro diz quanto esperar.
                if "429" in ultimo_erro or "RESOURCE_EXHAUSTED" in ultimo_erro:
                    time.sleep(_segundos_de_espera(ultimo_erro))
                    continue

                # Sobrecarga do provedor. É transitória e não tem nada a ver
                # com a nossa taxa — tratá-la como falha definitiva foi o que
                # perdeu 42 de 54 chamadas de uma bateria inteira. Backoff
                # exponencial com teto.
                if any(c in ultimo_erro for c in ("503", "UNAVAILABLE", "500", "INTERNAL")):
                    time.sleep(min(4 * (2 ** tentativa), 60))
                    continue

                return "", 0, 0, time.perf_counter() - t0, ultimo_erro

            dt = time.perf_counter() - t0
            u = r.usage_metadata
            return (
                r.text or "",
                int(u.prompt_token_count or 0),
                int(u.candidates_token_count or 0),
                dt,
                "",
            )

        return "", 0, 0, 0.0, ultimo_erro

    # -- modos ------------------------------------------------------------
    def responder(self, pergunta: str, resultados, resolver_versoes: bool = True) -> Resposta:
        """Resposta ancorada nos trechos recuperados.

        `resolver_versoes` liga o tratamento de empilhamento: 7,4% dos
        chunks do corpus trazem várias gerações do mesmo dispositivo
        coladas, e só a última vale. Sem resolver, o modelo recebe texto
        revogado com aparência de vigente — e o parâmetro existe para
        que o harness meça o custo disso, em vez de assumir.
        """
        from . import versioning

        blocos = []
        for i, r in enumerate(resultados, 1):
            texto = versioning.resolver(r.texto) if resolver_versoes else r.texto
            blocos.append(f"[trecho {i}] {r.citacao}\n{texto}")
        contexto = "\n\n".join(blocos)

        conteudo = f"TRECHOS NORMATIVOS:\n\n{contexto}\n\nPERGUNTA: {pergunta}"
        bruto, ent, sai, dt, erro = self._chamar(INSTRUCAO, conteudo)
        d = _extrair_json(bruto)

        return Resposta(
            pergunta=pergunta,
            texto=str(d.get("resposta", "")),
            citacoes=[c for c in d.get("citacoes", []) if isinstance(c, dict)],
            suficiente=bool(d.get("suficiente", True)),
            contexto_usado=[b.split("\n", 1)[1] for b in blocos],
            tokens_entrada=ent,
            tokens_saida=sai,
            latencia_s=dt,
            modo="rag",
            versoes_resolvidas=resolver_versoes,
            erro=erro,
        )

    def responder_sem_contexto(self, pergunta: str) -> Resposta:
        """Baseline: o que o modelo diz de cabeça, sem recuperação."""
        bruto, ent, sai, dt, erro = self._chamar(INSTRUCAO_SEM_CONTEXTO, f"PERGUNTA: {pergunta}")
        d = _extrair_json(bruto)
        return Resposta(
            pergunta=pergunta,
            texto=str(d.get("resposta", "")),
            citacoes=[c for c in d.get("citacoes", []) if isinstance(c, dict)],
            suficiente=bool(d.get("suficiente", True)),
            contexto_usado=[],
            tokens_entrada=ent,
            tokens_saida=sai,
            latencia_s=dt,
            modo="sem_contexto",
            erro=erro,
        )
