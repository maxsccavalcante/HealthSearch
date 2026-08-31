import streamlit as st
import pandas as pd
import numpy as np

# HEALTHSEARCH
# Motor de Busca Híbrido
# BM25 + Busca Semântica + RRF

# 1. CORPUS MÉDICO

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

# INTERFACE INICIAL

st.set_page_config(
    page_title="HealthSearch",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 HealthSearch")
st.subheader("Motor de Busca Híbrido — BM25 + Busca Semântica + RRF")

st.write(
    "Sistema de recuperação de informação médica utilizando "
    "busca léxica, busca semântica e fusão de rankings."
)

st.dataframe(df)
