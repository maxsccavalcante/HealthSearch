import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.set_page_config(
    page_title="HealthSearch",
    page_icon="🏥",
    layout="wide"
)


# ==========================================
# STOPWORDS EM PORTUGUÊS
# ==========================================

nltk.download("stopwords")

STOPWORDS_PT = set(
    stopwords.words("portuguese")
)


# ==========================================
# PRÉ-PROCESSAMENTO
# ==========================================

def preprocessar_texto(texto):

    texto = texto.lower()

    texto = re.sub(
        r"[^a-záàâãéêíóôõúç0-9\s-]",
        " ",
        texto
    )

    tokens = texto.split()

    tokens = [
        token
        for token in tokens
        if token not in STOPWORDS_PT
    ]

    return tokens

df = pd.DataFrame(documentos)


# ==========================================
# FUNÇÃO RRF
# ==========================================

def calcular_rrf(
    ranking_bm25,
    ranking_semantico,
    alpha,
    k_rrf=60
):

    scores_rrf = np.zeros(len(df))

    for rank, indice in enumerate(
        ranking_bm25,
        start=1
    ):
        scores_rrf[indice] += (
            alpha *
            (1 / (k_rrf + rank))
        )

    for rank, indice in enumerate(
        ranking_semantico,
        start=1
    ):
        scores_rrf[indice] += (
            (1 - alpha) *
            (1 / (k_rrf + rank))
        )

    return scores_rrf


# ==========================================
# CORPUS MÉDICO
# ==========================================

documentos = [

    {
        "id": "Doc 1",
        "titulo": "Protocolo Emergência ECG",
        "conteudo": (
            "Pacientes com dor precordial aguda e suspeita de "
            "síndrome coronariana devem realizar eletrocardiograma "
            "CÓD-ECG-12D em até 10 minutos."
        )
    },

    {
        "id": "Doc 2",
        "titulo": "Guia de Farmacologia Cardíaca",
        "conteudo": (
            "O uso imediato de ácido acetilsalicílico e "
            "antiagregantes plaquetários reduz a mortalidade "
            "no infarto agudo do miocárdio."
        )
    },

    {
        "id": "Doc 3",
        "titulo": "Diretriz de Hipertensão Arterial",
        "conteudo": (
            "A crise hipertensiva severa requer administração "
            "de anti-hipertensivos venosos e monitoramento "
            "contínuo da pressão arterial na UTI."
        )
    },

    {
        "id": "Doc 4",
        "titulo": "Manual de AVC Isquêmico",
        "conteudo": (
            "O acidente vascular cerebral isquêmico agudo deve "
            "ser tratado com trombolíticos venosos em até quatro "
            "horas e meia do início dos sintomas."
        )
    },

    {
        "id": "Doc 5",
        "titulo": "Protocolo de Reanimação RCR",
        "conteudo": (
            "Parada cardiorrespiratória em adultos exige compressões "
            "torácicas contínuas de alta qualidade e desfibrilação "
            "precoce no código azul."
        )
    },

    {
        "id": "Doc 6",
        "titulo": "Procedimentos de UTI Geral",
        "conteudo": (
            "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias "
            "complexas, recomenda-se a monitorização cardíaca "
            "contínua por telemetria."
        )
    }
]


# ==========================================
# DATAFRAME
# ==========================================

df = pd.DataFrame(documentos)


# ==========================================
# PRÉ-PROCESSAR DOCUMENTOS
# ==========================================

df["tokens"] = df["conteudo"].apply(
    preprocessar_texto
)


# ==========================================
# CORPUS DO BM25
# ==========================================

corpus_bm25 = df["tokens"].tolist()

# ==========================================
# MODELO DE EMBEDDINGS
# ==========================================

modelo_embedding = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ==========================================
# EMBEDDINGS DOS DOCUMENTOS
# ==========================================

embeddings_documentos = modelo_embedding.encode(
    df["conteudo"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True
)

# ==========================================
# SIDEBAR — CONFIGURAÇÕES BM25
# ==========================================

st.sidebar.header("⚙️ Configurações BM25")

k1 = st.sidebar.slider(
    "k1 — Saturação da frequência",
    min_value=0.0,
    max_value=3.0,
    value=1.2,
    step=0.1
)

b = st.sidebar.slider(
    "b — Normalização do tamanho",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05
)

# ==========================================
# CONFIGURAÇÃO DO RRF
# ==========================================

st.sidebar.header("⚖️ Configurações RRF")

alpha = st.sidebar.slider(
    "α — Peso do BM25",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)


# ==========================================
# CRIAR MOTOR BM25
# ==========================================

bm25 = BM25Okapi(
    corpus_bm25,
    k1=k1,
    b=b
)


# ==========================================
# INTERFACE
# ==========================================

st.title("🏥 HealthSearch")

st.subheader(
    "Motor de Busca Híbrido — BM25 + Busca Semântica + RRF"
)

st.write(
    "Sistema de recuperação de informação médica utilizando "
    "busca léxica, busca semântica e fusão de rankings."
)


# ==========================================
# CAMPO DE PESQUISA
# ==========================================

st.header("🔎 Pesquisa Médica")

consulta = st.text_input(
    "Digite sua consulta:",
    placeholder="Ex.: infarto, ECG-12D, AVC..."
)


# ==========================================
# EXECUTAR BUSCA BM25
# ==========================================

if consulta:

    consulta_tokens = preprocessar_texto(
        consulta
    )

    # ======================================
# EMBEDDING DA CONSULTA
# ======================================

embedding_consulta = modelo_embedding.encode(
    [consulta],
    convert_to_numpy=True,
    normalize_embeddings=True
)

scores_semanticos = (
    embeddings_documentos @ embedding_consulta[0]
)

ranking_semantico = np.argsort(
    scores_semanticos
)[::-1]

# ======================================
# RANKING HÍBRIDO RRF
# ======================================

scores_rrf = calcular_rrf(
    ranking_bm25,
    ranking_semantico,
    alpha
)

ranking_rrf = np.argsort(
    scores_rrf
)[::-1]

resultados_rrf = df.iloc[
    ranking_rrf
].copy()

resultados_rrf["score_rrf"] = (
    scores_rrf[ranking_rrf]
)

resultados_rrf["rank_rrf"] = range(
    1,
    len(resultados_rrf) + 1
)

resultados_rrf = resultados_rrf[
    [
        "rank_rrf",
        "id",
        "titulo",
        "score_rrf"
    ]
]

with tab3:

    st.subheader(
        "🔀 Ranking Híbrido — RRF"
    )

    st.dataframe(
        resultados_rrf,
        use_container_width=True
    )

    scores_bm25 = bm25.get_scores(
        consulta_tokens
    )

    ranking_bm25 = np.argsort(
        scores_bm25
    )[::-1]

resultados_semanticos = df.iloc[
    ranking_semantico
].copy()

resultados_semanticos["score_semantico"] = (
    scores_semanticos[ranking_semantico]
)

resultados_semanticos["rank_semantico"] = range(
    1,
    len(resultados_semanticos) + 1
)

resultados_semanticos = resultados_semanticos[
    [
        "rank_semantico",
        "id",
        "titulo",
        "score_semantico"
    ]
]

# ======================================
# MATRIZ COMPARATIVA
# ======================================

rank_bm25 = np.empty(
    len(df),
    dtype=int
)

rank_semantico = np.empty(
    len(df),
    dtype=int
)

rank_rrf = np.empty(
    len(df),
    dtype=int
)


for posicao, indice in enumerate(
    ranking_bm25,
    start=1
):
    rank_bm25[indice] = posicao


for posicao, indice in enumerate(
    ranking_semantico,
    start=1
):
    rank_semantico[indice] = posicao


for posicao, indice in enumerate(
    ranking_rrf,
    start=1
):
    rank_rrf[indice] = posicao

# ======================================
# MATRIZ COMPARATIVA
# ======================================

rank_bm25 = np.empty(
    len(df),
    dtype=int
)

rank_semantico = np.empty(
    len(df),
    dtype=int
)

rank_rrf = np.empty(
    len(df),
    dtype=int
)


for posicao, indice in enumerate(
    ranking_bm25,
    start=1
):
    rank_bm25[indice] = posicao


for posicao, indice in enumerate(
    ranking_semantico,
    start=1
):
    rank_semantico[indice] = posicao


for posicao, indice in enumerate(
    ranking_rrf,
    start=1
):
    rank_rrf[indice] = posicao

matriz_comparativa = df[
    ["id", "titulo"]
].copy()

matriz_comparativa["Rank BM25"] = (
    rank_bm25
)

matriz_comparativa["Rank Semântico"] = (
    rank_semantico
)

matriz_comparativa["Rank RRF"] = (
    rank_rrf
)

with tab4:

    st.subheader(
        "📋 Matriz Comparativa"
    )

    st.dataframe(
        matriz_comparativa,
        use_container_width=True
    )
st.dataframe(
    matriz_comparativa,
    use_container_width=True
)

with tab2:

    st.subheader(
        "🧠 Ranking Semântico"
    )

    st.dataframe(
        resultados_semanticos,
        use_container_width=True
    )

    # ======================================
    # RESULTADOS
    # ======================================

    resultados_bm25 = df.iloc[
        ranking_bm25
    ].copy()


    # Score
    resultados_bm25["score_bm25"] = (
        scores_bm25[ranking_bm25]
    )


    # Ranking
    resultados_bm25["rank_bm25"] = range(
        1,
        len(resultados_bm25) + 1
    )


    # Organizar colunas
    resultados_bm25 = resultados_bm25[
        [
            "rank_bm25",
            "id",
            "titulo",
            "score_bm25"
        ]
    ]


    # ======================================
    # MOSTRAR RESULTADOS
    # ======================================

 with tab1:

    st.subheader(
        "📊 Ranking Léxico — BM25"
    )

    st.dataframe(
        resultados_bm25,
        use_container_width=True
    )
