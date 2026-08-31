import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords

# HEALTHSEARCH

st.set_page_config(
    page_title="HealthSearch",
    page_icon="🏥",
    layout="wide"
)

# STOPWORDS

nltk.download("stopwords")

STOPWORDS_PT = set(stopwords.words("portuguese"))

# PRÉ-PROCESSAMENTO

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

# CORPUS MÉDICO

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


df = pd.DataFrame(documentos)

# APLICAR PRÉ-PROCESSAMENTO

df["tokens"] = df["conteudo"].apply(preprocessar_texto)

# INTERFACE

st.title("🏥 HealthSearch")

st.subheader(
    "Motor de Busca Híbrido — BM25 + Busca Semântica + RRF"
)

st.write(
    "Sistema de recuperação de informação médica utilizando "
    "busca léxica, busca semântica e fusão de rankings."
)

# TESTE DO PRÉ-PROCESSAMENTO

st.header("🔎 Teste do Pré-processamento")

for _, documento in df.iterrows():

    st.write(
        f"**{documento['id']} — {documento['titulo']}**"
    )

    st.write(documento["tokens"])
