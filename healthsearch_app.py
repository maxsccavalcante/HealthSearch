import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi


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

    scores_bm25 = bm25.get_scores(
        consulta_tokens
    )

    ranking_bm25 = np.argsort(
        scores_bm25
    )[::-1]


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

    st.subheader("📊 Ranking BM25")

    st.dataframe(
        resultados_bm25,
        use_container_width=True
    )
