import streamlit as st

# Page name
st.set_page_config(page_title="Análise Transportes na RIDE-DF", page_icon="🚘")
st.set_page_config(layout="wide")


st.title("Análise Multivariada: Transportes na RIDE-DF")
st.write("""
Bem-vindo ao aplicativo de análise multivariada focado na área de Transportes na Região Integrada de Desenvolvimento do Distrito Federal (RIDE-DF). 
Aqui você encontrará diversas análises relacionadas a acidentes de trânsito, infraestrutura viária e indicadores socioeconômicos dos municípios da RIDE-DF.
""")

st.divider()

st.markdown("<h4>Navegação</h4>", unsafe_allow_html=True)
st.markdown("""
Este aplicativo é dividido em cinco seções principais, cada uma focada em uma hipótese diferente relacionada aos transportes na RIDE-DF. Utilize o menu lateral ou os botões abaixo para navegar entre as seções:
""")

col1, col2, col3, col4 = st.columns(4, gap="small", border=True)
with col1:
    st.page_link("pages/acidentes_frota_nova.py", label="Frotas Mais Novas Geram Menos Acidentes", icon="🆕")
with col2:
    st.page_link("pages/acidentes_frota.py", label="Acidentes e Frota de Veículos", icon="🚗")
with col3:
    st.page_link("pages/acidentes_pib.py", label="Acidentes e o PIB dos Municípios", icon="💰")
with col4:
    st.page_link("pages/acidentes_infra.py", label="Acidentes e Infraestrutura Viária", icon="🚧")

st.divider()

st.markdown("""<h4>Sobre o Aplicativo</h4>
            
Este aplicativo foi desenvolvido para colocar em prática **técnicas de análise multivariada** aprendidas na matéria de análise multivariada do curso de **Ciência de Dados e Inteligência Artificial** do **IESB**, ministrada pela professora **Nátalia Evangelista**. As técnicas utilizadas incluem **Análise de Componentes Principais (PCA), Análise de Correspondência Múltipla (MCA) e clustering**, aplicadas ao contexto dos transportes na RIDE-DF.""", unsafe_allow_html=True)

st.markdown("""
<h5>Autores</h5>
<ul>
    <li><strong>Enzo Rodrigues Teixeira de Andrade</strong></li>
    <li><strong>Felipe Toledo Neves</strong></li>
    <li><strong>Luca Adriano Melo Mendonça Soares</strong></li>
    <li><strong>Marley Abe Silva</strong></li>
    <li><strong>Maycon Moriy Abe Machado</strong></li>
    <li><strong>Robson Ricardo Leite da Silva</strong></li>
    <li><strong>Victor Kauan Moreno de Brito</strong></li>
    <li><strong>Vinicius de Paula Ribeiro</strong></li>
</ul>
""", unsafe_allow_html=True)