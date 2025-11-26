import streamlit as st

st.title("Página inicial - Análise multivariada: Análise da área de Transportes na RIDE-DF")
st.write("""
Bem-vindo ao aplicativo de análise multivariada focado na área de Transportes na Região Integrada de Desenvolvimento do Distrito Federal (RIDE-DF). 
Aqui você encontrará diversas análises relacionadas a acidentes de trânsito, infraestrutura viária e indicadores socioeconômicos dos municípios da RIDE-DF.
""")

st.write("""
### Navegação
Utilize os botões a seguir ou o menu lateral para navegar entre as diferentes seções do aplicativo:
""")

st.page_link("pages/acidentes_pib.py", label="Acidentes e o PIB dos municípios", icon="💰")
st.page_link("pages/acidentes_infra.py", label="Acidentes e infraestrutura viária", icon="🚧")
st.page_link("pages/acidentes_frota.py", label="Acidentes e frota de veículos", icon="🚗")


st.write("""
### Sobre o Aplicativo
Este aplicativo foi desenvolvido para colocar em prática **técnicas de análise multivariada** aprendidas na matéria de análise multivariada do curso de **Ciência de Dados e Inteligência Artificial** do **IESB**, ministrada pela professora **Nátalia Evangelista**. As técnicas utilizadas incluem **Análise de Componentes Principais (PCA), Análise de Correspondência Múltipla (MCA) e clustering**, aplicadas ao contexto dos transportes na RIDE-DF.
         
### Autores
- **Enzo**
- **Felipe Toledo Neves**
- **Luca Adriano Melo Mendonça Soares**
- **Marley**
- **Maycon**
- **Robson Ricardo Silva Leite**
- **Victor Kauan**
- **Vinicius de Paula**
""")