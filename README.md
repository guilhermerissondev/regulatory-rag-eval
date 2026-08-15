# Harness de Avaliação para RAG Regulatório

**Num corpus de 294 normativos do BACEN sobre Pix, 44,6% estão revogados. O sistema de busca entrega norma morta ao modelo em 80% das perguntas — e nenhuma métrica de similaridade acusa isso.**

Este projeto não é um chatbot sobre regulação. É a medição de um — e a parte medida não precisa de LLM nenhum: o que a recuperação acerta, e o que o contexto entrega de podre antes mesmo do modelo abrir a boca.

---

## Por que este projeto existe

Montar um RAG hoje é trabalho de uma tarde. O difícil é responder se dá para confiar nele — e essa pergunta quase nunca aparece nos portfólios, porque ela exige conjunto dourado, verdade de referência e disposição de publicar o número ruim.

Em domínio regulado a pergunta deixa de ser acadêmica. Uma resposta fluente que cita a Resolução revogada ano passado tem a mesma aparência da resposta certa.

## O corpus

294 normativos coletados da API do BACEN — 6,5 milhões de caracteres de texto integral. Os dois endpoints usados não são documentados: o de busca apareceu numa pesquisa, o de texto integral saiu da inspeção das chamadas de rede do próprio site.

| | |
|---|---|
| Normativos | 294 |
| **Revogados** | **131 (44,6%)** |
| Chunks (por artigo) | 6.360 |
| Chunks de norma revogada | 37,2% |
| Chunks de artigo revogado dentro de norma vigente | 2,4% |
| Tipos | Resolução BCB, Resolução CMN, Instrução Normativa, Circular, Carta Circular, Resolução Conjunta |

## Três descobertas que mudaram o desenho

Todas vieram de ler o texto bruto antes de modelar. Duas derrubaram premissas minhas.

### 1. O texto empilha todas as versões históricas

Eu supunha que o risco fosse o sistema citar redação superada. Fui verificar: o BACEN publica texto **consolidado**, com marcador de atribuição. Premissa errada — mas o que encontrei no lugar é pior.

O documento mantém **todas as gerações de cada dispositivo, coladas em sequência**. O art. 39 do Regulamento do Pix:

```
II - houver problemas na identificação do usuário recebedor.
II - ... ; ou   (Redação dada, a partir de 1º/2/2022, pela Res. BCB nº 181)
II - ... ;      (Redação dada pela Res. BCB nº 402, de 22/7/2024.)
```

Três versões do mesmo inciso no mesmo trecho. Só a última vale. O modelo não tem como saber qual, e a tendência é citar a primeira — a mais antiga.

**468 dos 6.360 chunks (7,4%)** têm empilhamento genuíno, com até **sete versões** do mesmo dispositivo. O módulo [`src/versioning.py`](src/versioning.py) resolve pela regra que o próprio documento impõe: dentro de um artigo, a última ocorrência é a vigente. O corpus encolhe 7,1%.

Nota sobre esse número: a primeira contagem que fiz deu 17,7%, contando qualquer rótulo repetido. Estava inflada — lista aninhada legítima também repete rótulo. A detecção estrita exige marcador de atribuição em pelo menos uma das cópias, e dá 7,4%.

### 2. Numeração de artigo colide dentro do mesmo documento

A Resolução BCB nº 1 tem **dois `Art. 1º`**: um na resolução, outro no Regulamento anexo, separados por 20 mil caracteres. Sem distinguir a parte, "art. 3" casa com dois textos diferentes e a auditoria de citação passa a comparar coisas distintas achando que são a mesma.

Cada chunk carrega a parte do documento, e a citação sai como `Resolução BCB nº 1, Regulamento anexo, art. 3`.

### 3. Artigos revogados continuam impressos

151 chunks trazem `(Revogado, a partir de..., pela Resolução...)` no corpo do texto. A norma está em vigor, o artigo não, e o texto segue lá. É o caso mais traiçoeiro dos três: a citação confere com o documento oficial e ainda assim está errada.

## Recuperação

Chunking por artigo, não por janela fixa. O motivo principal não é qualidade — é que **segmentar por artigo é o que torna a avaliação possível**. Sem a etiqueta do artigo no chunk, não há como cruzar a citação da resposta com o estado de vigência do dispositivo.

Conjunto dourado de 18 perguntas escritas à mão, cada uma ancorada em dispositivo verificado. Não foi gerado por LLM de propósito: pergunta escrita a partir de um chunk pelo mesmo tipo de modelo que depois vai respondê-la mede circularidade, não capacidade.

| Estratégia | MRR | recall@1 | recall@3 | recall@5 | sem nenhum acerto |
|---|---|---|---|---|---|
| BM25 | 0,361 | 0,133 | 0,600 | 0,733 | 26,7% |
| Densa (E5 multilíngue) | 0,478 | 0,333 | 0,533 | 0,800 | 13,3% |
| **Híbrida (RRF)** | **0,644** | **0,533** | **0,800** | 0,800 | 20,0% |

A fusão é feita sobre **posição**, não sobre score: similaridade de cosseno e score BM25 vivem em escalas sem relação, e somá-los deixa o resultado à mercê de qual dos dois produz números maiores por acaso.

### As três perguntas que a recuperação errou

| Pergunta | Onde o alvo caiu |
|---|---|
| q04 — "data de entrada em vigor da Resolução que instituiu o Pix" | posição **171** |
| q09 — "quem se sujeita ao Regulamento do Pix" | posição **18** |
| q14 — "limites de valor para pessoas físicas" | posição **54** |

A q04 é instrutiva: **toda** norma tem um artigo de entrada em vigor, então o corpus contém centenas de trechos quase idênticos e o discriminante da pergunta não é o que o recuperador indexa.

A q09 é a contaminação em estado puro — **as posições 1, 2 e 4 são normas revogadas**, e o artigo certo está na 18ª.

## Contaminação do contexto

O número central do projeto. Com os 8 trechos que vão para o modelo:

| | |
|---|---|
| Perguntas cujo contexto contém **norma revogada** | **80,0%** |
| Perguntas cujo contexto contém artigo revogado | 26,7% |
| Trechos com versões empilhadas | 10,8% |

Quatro de cada cinco perguntas entregam direito morto ao modelo, misturado ao vivo, sem nenhuma marcação. Um RAG comum não tem como saber — e o usuário, muito menos.

## Geração — implementada, não medida

O bloco de geração está escrito e funcional: [`src/generation.py`](src/generation.py) faz a chamada com saída estruturada, exige citação por afirmação, tem limitador de taxa e retry com backoff, e mede tokens, latência e custo por pergunta. O harness compara três modos — contexto com versões resolvidas, contexto bruto, e o modelo respondendo sem recuperação nenhuma.

**Os números não estão aqui porque a camada gratuita não sustentou a bateria.** Quatro tentativas, três causas diferentes:

| Tentativa | Falha | Correção aplicada |
|---|---|---|
| 1ª | 32/54 com 429 — 5 req/min estourado | limitador de taxa próprio |
| 2ª | 30 min em retry, sem terminar | limite baixado de 5 para 4/min |
| 3ª | 54/54 com 429 — **a cota é por modelo**, e o `gemini-2.5-flash` já estava esgotado | troca para `gemini-flash-latest` |
| 4ª | 42/54 com 503 — sobrecarga do provedor | backoff exponencial para 503/500 |

Cada correção é código que ficou, e a última cobre a lacuna real: um harness que trata erro transitório do provedor como falha definitiva perde a bateria inteira. Com chave paga, `python -m scripts.evaluate` roda os três modos sem alteração nenhuma.

**Publicar sem esses números é decisão consciente, não descuido.** Métrica calculada sobre resposta que nunca chegou é pior que métrica nenhuma, porque parece resultado — e uma rodada anterior deste projeto produziu exatamente isso antes de ser descartada.

As duas camadas acima **não usam LLM** e são as que sustentam o argumento: recuperação medida contra conjunto dourado, e integridade do corpus medida contra os metadados de vigência. É a metade do problema que a maioria dos projetos de RAG pula.

## Rodando

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
```

A geração usa a API do Gemini. Defina a chave no ambiente — ela nunca aparece no código:

```bash
setx GEMINI_API_KEY "sua-chave"
```

Coleta do corpus, chunking e índice (alguns minutos em CPU):

```bash
python -m scripts.build_index
```

Harness completo. Com `--so-recuperacao` roda tudo que não gasta token:

```bash
python -m scripts.evaluate --so-recuperacao
```

Os dois blocos que não gastam token são o núcleo do projeto. Para incluir a geração — que exige chave com cota real — rode sem a flag:

```bash
python -m scripts.evaluate
```

App com a auditoria à vista:

```bash
streamlit run app/streamlit_app.py
```

Testes:

```bash
python -m pytest tests/ -q
```

## Estrutura

```
src/
  config.py       parâmetros de corpus, busca, geração e harness
  corpus.py       ingestão da API do BACEN
  chunking.py     segmentação por artigo, com parte e vigência
  versioning.py   resolução de versões empilhadas
  retrieval.py    busca densa, léxica e fusão por RRF
  metrics.py      recall@k, MRR e auditoria de vigência
  generation.py   geração com citação obrigatória e limitador de taxa
scripts/
  build_index.py  corpus -> chunks -> embeddings
  evaluate.py     harness de três blocos
data/golden.json  conjunto dourado, escrito à mão
app/              assistente com selo de vigência por trecho
tests/            47 testes
reports/          tabelas geradas pelo harness
```

## Escolhas técnicas

**Embeddings locais em CPU.** `multilingual-e5-small`, ~7 chunks/s, 15 minutos para o corpus. Sem GPU e sem custo de API na parte de recuperação — que é onde estão duas das três camadas de avaliação.

**O prefixo do E5 importa.** O modelo exige `query:` na consulta e `passage:` na passagem. Omitir degrada a recuperação **sem emitir erro nenhum** — o tipo de bug que só aparece na métrica.

**Saída estruturada na geração.** Um parágrafo com "conforme a Resolução BCB nº 1" no meio é impossível de auditar programaticamente. Exigindo lista de citações com norma e artigo em campos separados, cada afirmação vira algo verificável. A auditoria de vigência só existe por causa disso.

**Limitador de taxa próprio.** A camada gratuita permite 5 requisições por minuto. Mirar exatamente no teto deixa cada chamada na borda da janela deslizante: a primeira execução passou 30 minutos em retry de 429 sem terminar, e uma anterior teve 32 de 54 chamadas falhando — com métricas calculadas sobre respostas vazias, que pareciam resultado. O harness roda a 4/min e descarta rodada com erro.

## Limites

**A geração não foi medida.** O código está pronto e testado, mas a camada gratuita da API não sustentou a bateria de 54 chamadas. Ver a seção acima; com chave paga não há alteração de código.

**O conjunto dourado tem 18 perguntas.** É pouco para intervalo de confiança estreito, e é o que dá para ancorar à mão com honestidade. Ampliar exige mais leitura de norma, não mais compute.

**O corpus tem ruído.** A busca por "pix" trouxe a Resolução BCB nº 340, que é o Regimento Interno do Banco Central — 383 chunks de assunto alheio. Ficaram: corpus real tem ruído, e eles servem de distratores honestos.

**A fundamentação é heurística.** Sobreposição de n-gramas entre resposta e contexto não distingue paráfrase de invenção. Serve como triagem, e por isso anda junto com a auditoria de citação, que é verificação exata.

**Vigência é medida contra o corpus, não contra o mundo.** Se uma norma foi revogada depois da coleta, o harness não sabe. Em produção isso exigiria reingestão periódica — o mesmo problema de drift dos modelos preditivos, com outra roupa.

## Fonte

[Busca de normas do Banco Central do Brasil](https://www.bcb.gov.br/estabilidadefinanceira/buscanormas). Coleta automatizada pelo pipeline; nenhum dado redistribuído neste repositório.
