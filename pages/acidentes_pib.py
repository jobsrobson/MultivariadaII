# importações e configuração do path

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/toledo-cia/Documents/Projetos/MultivariadaII')

from app import fetch_data
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import unicodedata
import plotly.express as px
import plotly.graph_objects as go

# funções auxiliares
def normalizar_texto(texto):
    if not isinstance(texto, str):
        return str(texto)
    nfkd = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).upper().strip()


# =========================  Carregamento e Tratamento dos Dados  =========================
df_acidentes = fetch_data("""
    SELECT 
        pesid, municipio, uf
    FROM public.acidente_transito
""")

df_acidentes_count = df_acidentes.groupby(['uf', 'municipio']).size().reset_index(name='total_acidentes')

# Normalizar
df_acidentes_count['chave_municipio'] = df_acidentes_count['municipio'].apply(normalizar_texto)
df_acidentes_count['uf'] = df_acidentes_count['uf'].astype(str).str.upper().str.strip()


df_populacao = fetch_data("""
    SELECT 
        "CO_MUNICIPIO", sum("TOTAL") as populacao_total
    FROM public."Censo_20222_Populacao_Idade_Sexo"
    GROUP BY 1
""")

df_populacao.rename(columns={'CO_MUNICIPIO': 'codigo_ibge_municipio'}, inplace=True)
df_populacao['codigo_ibge_municipio'] = df_populacao['codigo_ibge_municipio'].astype(str)

df_pib = fetch_data("""
                    SELECT 
                        ano_pib, 
                        codigo_municipio_dv, 
                        vl_pib_per_capta
                    FROM public.pib_municipios
                    WHERE 
                        ano_pib = (SELECT MAX(ano_pib) FROM public.pib_municipios)
                    """)

df_pib.rename(columns={'codigo_municipio_dv': 'codigo_ibge_municipio'}, inplace=True)
df_pib['codigo_ibge_municipio'] = df_pib['codigo_ibge_municipio'].astype(str)
df_pib['vl_pib_per_capta'] = pd.to_numeric(df_pib['vl_pib_per_capta'], errors='coerce')

df_municipio_base = fetch_data("""
    SELECT 
        codigo_municipio_dv, 
        nome_municipio, 
        cd_uf
    FROM public.municipio
""")

df_municipio_base.rename(columns={'codigo_municipio_dv': 'codigo_ibge_municipio', 'nome_municipio': 'municipio_nome'}, inplace=True)

# Tratamento da Chave de Nome
df_municipio_base['chave_municipio'] = df_municipio_base['municipio_nome'].apply(normalizar_texto)
df_municipio_base['codigo_ibge_municipio'] = df_municipio_base['codigo_ibge_municipio'].astype(str)

# --- CORREÇÃO DO ERRO ANTERIOR AQUI ---
# Converter cd_uf para numérico antes de fazer o mapeamento
df_municipio_base['cd_uf'] = pd.to_numeric(df_municipio_base['cd_uf'], errors='coerce')

# Mapeamento UF (Agora vai funcionar pois os dados estarão como números)
map_uf = {
    11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
    21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA',
    31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP',
    41: 'PR', 42: 'SC', 43: 'RS',
    50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'
}
df_municipio_base['uf'] = df_municipio_base['cd_uf'].map(map_uf)

df_merged = pd.merge(
        df_acidentes_count,
        df_municipio_base,
        on=['uf', 'chave_municipio'],
        how='inner'
    )

if df_merged.shape[0] > 0:
    # Merge 2: + PIB
    df_merged = pd.merge(
        df_merged,
        df_pib[['codigo_ibge_municipio', 'vl_pib_per_capta']],
        on='codigo_ibge_municipio',
        how='inner'
    )

    # Merge 3: + População
    df_merged = pd.merge(
        df_merged,
        df_populacao[['codigo_ibge_municipio', 'populacao_total']],
        on='codigo_ibge_municipio',
        how='inner'
    )

    # Filtrar
    df_analise = df_merged[
        (df_merged['populacao_total'] > 0) &
        (df_merged['vl_pib_per_capta'].notna())
    ].copy()

df_analise['taxa_acidentes_100k'] = (df_analise['total_acidentes'] / df_analise['populacao_total']) * 100000

# Ordenar para ver quem tem a maior taxa
df_exibicao = df_analise[['uf', 'chave_municipio', 'populacao_total', 'vl_pib_per_capta', 'total_acidentes', 'taxa_acidentes_100k']]

corr_coef, p_value = stats.pearsonr(df_analise['vl_pib_per_capta'], df_analise['taxa_acidentes_100k'])

interpretacao = ""
if abs(corr_coef) < 0.3: interpretacao = "Fraca"
elif abs(corr_coef) < 0.7: interpretacao = "Moderada"
else: interpretacao = "Forte"

tipo = "Positiva" if corr_coef > 0 else "Negativa"

# Passo A: Criar Categorias de Riqueza (Discretização)
# Dividir o PIB em 3 grupos: Baixo (0-33%), Médio (33-66%), Alto (66-100%)
df_analise['faixa_pib'] = pd.qcut(df_analise['vl_pib_per_capta'], q=3, labels=['Baixa Renda', 'Média Renda', 'Alta Renda'])

# Passo B: Separar os grupos
grupo_baixa = df_analise[df_analise['faixa_pib'] == 'Baixa Renda']['taxa_acidentes_100k']
grupo_media = df_analise[df_analise['faixa_pib'] == 'Média Renda']['taxa_acidentes_100k']
grupo_alta = df_analise[df_analise['faixa_pib'] == 'Alta Renda']['taxa_acidentes_100k']

# Passo C: Rodar ANOVA
f_stat, p_value_anova = stats.f_oneway(grupo_baixa, grupo_media, grupo_alta)

X = df_analise[['vl_pib_per_capta', 'taxa_acidentes_100k']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Criar 3 Clusters (Grupos)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_analise['cluster'] = kmeans.fit_predict(X_scaled)

# =========================  Dashboard  =========================
st.markdown("<h2>Hipótese 3: Perfil Econômico dos Municípios<br>e os Acidentes de Trânsito</h2>" , unsafe_allow_html=True)
st.markdown("""
O perfil econômico dos municípios também pode influenciar os padrões de acidentes. Contextos de maior vulnerabilidade social podem estar associados a menor acesso a veículos seguros, infraestrutura deficiente e maior exposição a riscos no trânsito. Esta hipótese pretende explorar como o Produto Interno Bruto (PIB) dos municípios se relaciona com os índices de acidentes.
""")


st.info(f"Municípios agrupados: {df_acidentes_count.shape[0]} municípios.")

st.markdown("<br>", unsafe_allow_html=True)

with st.container(border=True):

    st.markdown("<b>Top 5 Municípios da RIDE-DF com Maior Taxa de Acidentes</b>", unsafe_allow_html=True)

    # Renomear colunas para exibição
    df_exibicao_display = df_exibicao.rename(columns={
        'uf': 'UF',
        'chave_municipio': 'Município',
        'populacao_total': 'População Total',
        'vl_pib_per_capta': 'PIB per Capita (R$)',
        'total_acidentes': 'Total de Acidentes',
        'taxa_acidentes_100k': 'Taxa de Acidentes por 100 mil hab.'
    })

    st.dataframe(df_exibicao_display.sort_values(by='Taxa de Acidentes por 100 mil hab.', ascending=False).head(5))


with st.container(border=True):
    st.markdown("<b>Correlação entre PIB per Capita e Taxa de Acidentes por 100 mil Habitantes</b>", unsafe_allow_html=True)

    fig = px.scatter(
        df_analise,
        x='vl_pib_per_capta',
        y='taxa_acidentes_100k',
        text='chave_municipio',
        labels={
            'vl_pib_per_capta': 'PIB per Capita (R$)',
            'taxa_acidentes_100k': 'Taxa de Acidentes por 100k habitantes'
        },
        title=" ",
        template='plotly_white'
    )

    # Adicionar linha de tendência
    fig.update_traces(marker=dict(size=10, opacity=0.7, color='steelblue'), textposition='top center')
    fig.add_trace(go.Scatter(
        x=df_analise['vl_pib_per_capta'],
        y=df_analise['vl_pib_per_capta'] * corr_coef + df_analise['taxa_acidentes_100k'].mean(),
        mode='lines',
        name='Tendência',
        line=dict(color='red', dash='dash')
    ))

    # Ajustar layout
    fig.update_layout(
        xaxis_title='PIB per Capita (R$)',
        yaxis_title='Taxa de Acidentes por 100k habitantes',
        title_font_size=16,
        title_x=0.5,
        showlegend=False,
        margin=dict(t=30, b=50, l=50, r=10)
        
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pd.DataFrame({
        "Métrica": ["Coeficiente de Correlação (Pearson)", "P-valor", "Interpretação"],
        "Valor": [f"{corr_coef:.4f}", f"{p_value:.4f}", f"Correlação {interpretacao} {tipo}"]
    }), hide_index=True)



st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h3>Análise de Variância (ANOVA) entre Faixas de Renda</h3>" , unsafe_allow_html=True)


with st.container(border=True):
    st.markdown("""
    Para entender melhor como o perfil econômico afeta a taxa de acidentes, realizamos uma Análise de Variância (ANOVA). Dividimos os municípios em três grupos com base no PIB per capita: Baixa Renda, Média Renda e Alta Renda. A ANOVA nos ajuda a determinar se há diferenças estatisticamente significativas nas taxas de acidentes entre esses grupos.
    """)
    st.dataframe(pd.DataFrame({
        "Métrica": ["Estatística F", "P-valor (ANOVA)"],
        "Valor": [f"{f_stat:.4f}", f"{p_value_anova:.4f}"]
    }), hide_index=True)

    if p_value_anova < 0.05:
        st.success("**Conclusão**: Há diferença estatisticamente significativa entre os grupos de renda.")
    else:
        st.error("**Conclusão**: NÃO há diferença significativa. A taxa de acidentes é parecida independente da faixa de renda.")


    # Visualização da ANOVA (Boxplot) com Plotly
    fig = px.box(
        df_analise,
        x='faixa_pib',
        y='taxa_acidentes_100k',
        color='faixa_pib',
        title='',
        labels={
            'faixa_pib': 'Faixa de Renda',
            'taxa_acidentes_100k': 'Acidentes por 100k hab.'
        },
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        xaxis_title='Faixa de Renda',
        yaxis_title='Acidentes por 100k hab.',
        showlegend=False,
        margin=dict(t=30, b=50, l=50, r=10)
    )
    st.plotly_chart(fig, use_container_width=True)




st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h3>Clustering dos Municípios por PIB e Taxa de Acidentes</h3>" , unsafe_allow_html=True)

with st.container(border=True):

    st.markdown("<h6>Agrupamento de Municípios - K-Means</h6>" , unsafe_allow_html=True)

    # Visualização do Clustering com Plotly
    fig = px.scatter(
        df_analise,
        x='vl_pib_per_capta',
        y='taxa_acidentes_100k',
        color='cluster',
        text='chave_municipio',
        title=' ',
        labels={
            'vl_pib_per_capta': 'PIB per Capita',
            'taxa_acidentes_100k': 'Taxa de Acidentes por 100k hab.',
            'cluster': 'Cluster'
        },
        template='plotly_white',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_traces(marker=dict(size=10, opacity=0.7), textposition='top center')
    fig.update_layout(
        xaxis_title='PIB per Capita',
        yaxis_title='Taxa de Acidentes por 100k hab.',
        title_font_size=16,
        title_x=0.5,
        margin=dict(t=30, b=50, l=50, r=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h6>Perfil Médio dos Grupos Encontrados</h6>" , unsafe_allow_html=True)

    # RENOMEAR COLUNAS PARA EXIBIÇÃO
    df_analise_display = df_analise.rename(columns={
        'vl_pib_per_capta': 'PIB per Capita (R$)',
        'taxa_acidentes_100k': 'Taxa de Acidentes por 100 mil hab.'
    })
    df_cluster_summary = df_analise_display.groupby('cluster')[['PIB per Capita (R$)', 'Taxa de Acidentes por 100 mil hab.']].mean().reset_index()
    st.dataframe(df_cluster_summary)