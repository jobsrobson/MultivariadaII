import streamlit as st


# --- 1. CONFIGURAÇÃO DA PÁGINA ---

st.set_page_config(
    page_title="Transporte RIDE-DF",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CONEXÃO COM BASE DE DADOS ---

conn = st.connection("postgres", type="sql")

# --- 3. DEFINIÇÃO DE FUNÇÕES ---

@st.cache_data(ttl=3600, show_spinner="Buscando dados do banco de dados...")
def fetch_data(query, connection=conn):
    """Função para buscar dados do banco de dados."""
    return connection.query(query)

st.header("Página inicial - Análise multivariada: Análise da área de Transportes na RIDE-DF")

