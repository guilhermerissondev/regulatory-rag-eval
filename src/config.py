"""Parâmetros do projeto.

O objeto de estudo aqui não é o chatbot — é a avaliação dele. Os
parâmetros que mais importam ficam na seção de harness, no fim.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

CORPUS_JSONL = DATA_RAW / "normativos.jsonl"
CHUNKS_PARQUET = DATA_PROCESSED / "chunks.parquet"
EMBEDDINGS_NPY = DATA_PROCESSED / "embeddings.npy"

# --------------------------------------------------------------------------
# Fonte
#
# A API de busca do BACEN devolve metadados; o texto integral vem de um
# segundo endpoint, com parâmetros `p1` (tipo) e `p2` (número). Os dois
# foram descobertos inspecionando as chamadas de rede do próprio site —
# não há documentação pública deles.
# --------------------------------------------------------------------------
API_BUSCA = "https://www.bcb.gov.br/api/search/app/normativos/buscanormativos"
API_TEXTO = "https://www.bcb.gov.br/api/conteudo/app/normativos/exibenormativo"

# Recorte do corpus: a família normativa do Pix. Coerente o bastante para
# que perguntas tenham resposta certa, e grande o bastante para que a
# recuperação seja um problema de verdade.
CONSULTA = "pix"
MAX_NORMATIVOS = 320
PAUSA_ENTRE_REQUISICOES = 0.4  # segundos; a API não documenta limite

# --------------------------------------------------------------------------
# Chunking e busca
# --------------------------------------------------------------------------
# Modelo multilíngue que roda em CPU. O E5 exige prefixos "query:" e
# "passage:" — omiti-los degrada a recuperação de forma silenciosa, que é
# o tipo de erro que este projeto existe para pegar.
MODELO_EMBEDDING = "intfloat/multilingual-e5-small"
PREFIXO_CONSULTA = "query: "
PREFIXO_PASSAGEM = "passage: "

MAX_CHARS_CHUNK = 1800
OVERLAP_CHARS = 200
TOP_K = 8

# Peso do denso na fusão híbrida com BM25 (0 = só léxico, 1 = só denso).
PESO_DENSO = 0.5

# --------------------------------------------------------------------------
# Geração
# --------------------------------------------------------------------------
# A cota da camada gratuita e POR MODELO. Depois de esgotar o
# gemini-2.5-flash numa bateria de avaliacao, o gemini-flash-latest
# continuava respondendo normalmente — trocar de modelo destrava, e o
# harness registra qual modelo de fato rodou.
MODELO_LLM = "gemini-flash-latest"
VARIAVEL_CHAVE = "GEMINI_API_KEY"
TEMPERATURA = 0.0
MAX_TOKENS_RESPOSTA = 1200

# A camada gratuita do Gemini permite 5 requisicoes por minuto por modelo.
# O harness respeita esse teto por conta propria: estourar a cota devolve
# resposta vazia, e metrica calculada sobre resposta que nunca chegou e
# pior que metrica nenhuma, porque parece resultado.
#
# O valor fica em 4, nao 5. Mirar exatamente no teto deixa cada chamada
# na borda da janela deslizante do provedor: a primeira execucao assim
# passou 30 minutos em retry de 429 sem terminar. Uma requisicao a menos
# por minuto custa segundos e evita a espera de 33s de cada recusa.
RPM_LIMITE = 4
# Duas tentativas, nao quatro. Cada 429 custa ~33s de espera; com cota
# realmente esgotada, insistir so transforma uma falha rapida em uma
# execucao de duas horas. Foi o que aconteceu antes de descobrir que a
# cota era por modelo.
# Cinco tentativas: cota esgotada falha rapido (o proprio 429 informa o
# retryDelay), mas 503 de sobrecarga do provedor e transitorio e merece
# backoff exponencial. Uma bateria inteira ja foi perdida por tratar 503
# como falha definitiva.
MAX_TENTATIVAS = 5

# --------------------------------------------------------------------------
# Harness
#
# As três perguntas que o projeto responde, nesta ordem de importância:
#
#   1. O trecho certo foi recuperado?      recall@k, MRR
#   2. A resposta se apoia no recuperado?  fundamentação
#   3. A norma citada ainda vale?          armadilha temporal
#
# A terceira é a que quase nenhum harness de RAG cobre, e é o modo de
# falha mais caro em domínio regulado: citar norma revogada ou artigo
# cuja redação foi superada por alteração posterior.
# --------------------------------------------------------------------------
KS_AVALIADOS = (1, 3, 5, 8, 10)
GOLDEN_JSON = ROOT / "data" / "golden.json"
