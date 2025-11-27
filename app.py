import streamlit as st


# --- 1. CONFIGURAÇÃO DA PÁGINA ---

pg = st.navigation([
    st.Page("pages/menu.py", title="Menu Principal", icon="🏠"),
    st.Page("pages/acidentes_frota_nova.py", title="Frotas Mais Novas Geram Menos Acidentes", icon="1️⃣"),
    st.Page("pages/acidentes_frota_potencia.py", title="Frotas Mais Potentes Geram Mais Acidentes", icon="2️⃣"),
    st.Page("pages/acidentes_infra.py", title="Infraestrutura Viária e Acidentes", icon="3️⃣"),
    st.Page("pages/acidentes_pib.py", title="PIB e Acidentes de Trânsito", icon="4️⃣"),
    st.Page("pages/acidentes_frota.py", title="Frota de Veículos e Acidentes", icon="5️⃣")
])


# --- 2. CONEXÃO COM BASE DE DADOS ---

conn = st.connection("postgres", type="sql")

# --- 3. DEFINIÇÃO DE FUNÇÕES ---

@st.cache_data(ttl=3600, show_spinner="Buscando dados do banco de dados...")
def fetch_data(query, connection=conn):
    """Função para buscar dados do banco de dados."""
    return connection.query(query)


pg.run()
