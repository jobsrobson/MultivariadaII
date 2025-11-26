import streamlit as st


# --- 1. CONFIGURAÇÃO DA PÁGINA ---

pg = st.navigation([
    st.Page("pages/menu.py", title="Menu Principal", icon="🏠"),
    st.Page("pages/acidentes_pib.py", title="Acidentes e o PIB dos municípios", icon="💰"),
    st.Page("pages/acidentes_infra.py", title="Acidentes e infraestrutura viária", icon="🚧"),
    st.Page("pages/acidentes_frota.py", title="Acidentes e frota de veículos", icon="🚗")
])


# --- 2. CONEXÃO COM BASE DE DADOS ---

conn = st.connection("postgres", type="sql")

# --- 3. DEFINIÇÃO DE FUNÇÕES ---

@st.cache_data(ttl=3600, show_spinner="Buscando dados do banco de dados...")
def fetch_data(query, connection=conn):
    """Função para buscar dados do banco de dados."""
    return connection.query(query)


pg.run()
