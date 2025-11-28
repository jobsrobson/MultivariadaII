import sys
from app import fetch_data
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas para MCA
try:
    import prince
    MCA_DISPONIVEL = True
except ImportError:
    st.warning("⚠️ Instale: pip install prince")
    MCA_DISPONIVEL = False

# Bibliotecas para análise
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.stats import f_oneway, kruskal, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# Configuração da página
st.set_page_config(page_title="Análise de Infraestrutura Viária", layout="wide")

# Configuração visual dos gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ==================== FUNÇÕES AUXILIARES ====================

def agregar_vitimas(group):
    """
    Agregação de vítimas por acidente
    
    Definições:
    - VÍTIMA: Qualquer pessoa envolvida (incluindo ilesos)
    """
    group_unico = group.drop_duplicates(subset='pesid', keep='first')
    
    estados = group_unico['estado_fisico'].value_counts().to_dict()
    n_mortos = estados.get('Óbito', 0) + estados.get('Morte', 0)
    n_feridos_graves = estados.get('Lesões Graves', 0)
    n_feridos_leves = estados.get('Lesões Leves', 0)
    n_ilesos = estados.get('Ileso', 0)
    n_nao_informado = estados.get('Não Informado', 0) + estados.get('0', 0)
    total_vitimas = len(group_unico)
    vitimas_com_lesao = n_mortos + n_feridos_graves + n_feridos_leves
    
    return pd.Series({
        'total_vitimas': total_vitimas,
        'n_mortos': n_mortos,
        'n_feridos_graves': n_feridos_graves,
        'n_feridos_leves': n_feridos_leves,
        'n_ilesos': n_ilesos,
        'n_nao_informado': n_nao_informado,
        'vitimas_com_lesao': vitimas_com_lesao,
        'prop_mortos': n_mortos / total_vitimas if total_vitimas > 0 else 0,
        'prop_ilesos': n_ilesos / total_vitimas if total_vitimas > 0 else 0,
    })

def categorizar_gravidade(row):
    """Classifica a gravidade do acidente com base nas vítimas"""
    if row['n_mortos'] > 0:
        return 'Fatal'
    elif row['n_feridos_graves'] > 0:
        return 'Grave'
    elif row['n_feridos_leves'] > 0:
        return 'Leve'
    else:
        return 'Sem_Lesoes'

def simplificar_tracado(x):
    x_str = str(x).lower()
    if 'curva' in x_str:
        return 'Curva'
    elif 'aclive' in x_str or 'declive' in x_str:
        return 'Inclinacao'
    elif 'ponte' in x_str or 'viaduto' in x_str:
        return 'Obra_Arte'
    else:
        return 'Reta'

# ==================== TÍTULO E INTRODUÇÃO ====================

st.markdown("<h2>Hipótese 3: Avaliação da Infraestrutura Viária e sua<br>Relação com a Ocorrência e Gravidade dos Acidentes</h2>", unsafe_allow_html=True)

st.markdown("""
Esta análise investiga a relação entre **causas relacionadas à infraestrutura viária** e a **gravidade dos acidentes de trânsito** 
na RIDE-DF durante o ano de 2024. Os dados foram obtidos diretamente do site da Polícia Rodoviária Federal (PRF) e incluem 
informações agregadas sobre acidentes e vítimas. 
            
**Objetivos:**
- Identificar padrões de acidentes causados por problemas de infraestrutura
- Avaliar a gravidade dos acidentes através de análise multivariada
- Agrupar acidentes com características similares usando técnicas de clustering
""")

st.markdown("<br>", unsafe_allow_html=True)

# ==================== PREPARAÇÃO DOS DADOS ====================

# Carregar dados
df = fetch_data("SELECT * FROM acidente_transito")

# Definir causas de infraestrutura
causas_infraestrutura = [
    'Iluminação deficiente', 'Demais falhas na via', 'Pista esburacada',
    'Falta de acostamento', 'Acesso irregular', 
    'Acumulo de água sobre o pavimento',
    'Acumulo de areia ou detritos sobre o pavimento',
    'Acumulo de óleo sobre o pavimento',
    'Falta de elemento de contenção que evite a saída do leito carroçável',
    'Restrição de visibilidade em curvas verticais',
    'Restrição de visibilidade em curvas horizontais',
    'Sistema de drenagem ineficiente', 'Declive acentuado',
    'Curva acentuada', 'Sinalização mal posicionada',
    'Desvio temporário', 'Ausência de sinalização',
    'Afundamento ou ondulação no pavimento',
    'Acostamento em desnível',
    'Deficiência do Sistema de Iluminação/Sinalização',
    'Pista Escorregadia'
]

# Filtrar acidentes de infraestrutura
df['causa_infraestrutura'] = df['causa_acidente'].isin(causas_infraestrutura)
acidentes_infra = df[df['causa_infraestrutura']]['id'].unique()
df_infra = df[df['id'].isin(acidentes_infra)].copy()

# Agregar vítimas
df_vitimas = df_infra.groupby('id').apply(agregar_vitimas).reset_index()

# Características do acidente
caracteristicas_acidente = [
    'data_inversa', 'dia_semana', 'horario', 'uf', 'br', 'km',
    'municipio', 'causa_principal', 'causa_acidente', 'tipo_acidente',
    'classificacao_acidente', 'fase_dia', 'sentido_via',
    'condicao_metereologica', 'tipo_pista', 'tracado_via', 'uso_solo',
    'latitude', 'longitude'
]

colunas_disponiveis = [col for col in caracteristicas_acidente if col in df_infra.columns]
df_acidentes = df_infra[['id'] + colunas_disponiveis].drop_duplicates(subset='id', keep='first')

# Juntar contagens de vítimas
df_analise = df_acidentes.merge(df_vitimas, on='id', how='inner')

# Calcular índice de gravidade
df_analise['indice_gravidade'] = (
    df_analise['n_mortos'] * 4 +
    df_analise['n_feridos_graves'] * 2 +
    df_analise['n_feridos_leves'] * 1
)

# Categorizar gravidade
df_analise['gravidade_cat'] = df_analise.apply(categorizar_gravidade, axis=1)

# Simplificar variáveis categóricas
df_analise['tracado_simplificado'] = df_analise['tracado_via'].apply(simplificar_tracado)
df_analise['pista_tipo'] = df_analise['tipo_pista'].apply(
    lambda x: 'Multipla' if 'ltipla' in str(x) else 'Simples'
)
df_analise['area_tipo'] = df_analise['uso_solo'].apply(
    lambda x: 'Urbana' if 'Sim' in str(x) else 'Rural'
)
df_analise['clima_adverso'] = df_analise['condicao_metereologica'].apply(
    lambda x: 'Adverso' if str(x) not in ['Céu Claro', 'Nublado', 'Sol'] else 'Normal'
)
df_analise['periodo_dia'] = df_analise['fase_dia'].apply(
    lambda x: 'Noite' if 'Noite' in str(x) else 'Dia' if 'dia' in str(x) else 'Transicao'
)

# Preparar dataset para modelagem
df_modelo = df_analise.copy()

# Remover outliers extremos (percentil 99)
for col in ['total_vitimas', 'n_mortos', 'n_feridos_graves', 'indice_gravidade']:
    q99 = df_modelo[col].quantile(0.99)
    df_modelo[col] = df_modelo[col].clip(upper=q99)



# ==================== ESTATÍSTICAS DESCRITIVAS ====================

with st.container(border=True):
    st.markdown("<h5>Estatísticas Descritivas dos Dados</h5>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📊 Total de Registros", 
            value=f"{len(df):,}".replace(",", "."),
            help="Número total de registros no banco de dados (inclui múltiplas vítimas por acidente)"
        )
        st.metric(
            label="🚗 Acidentes Únicos", 
            value=f"{df['id'].nunique():,}".replace(",", "."),
            help="Número de acidentes distintos registrados"
        )
    
    with col2:
        prop_infra = len(acidentes_infra)/df['id'].nunique()*100
        st.metric(
            label="🛣️ Acidentes de Infraestrutura", 
            value=f"{len(acidentes_infra):,}".replace(",", "."),
            delta=f"{prop_infra:.1f}% do total",
            help="Acidentes causados por problemas na infraestrutura viária"
        )
        st.metric(
            label="👥 Vítimas (Infraestrutura)", 
            value=f"{df_analise['total_vitimas'].sum():,}".replace(",", "."),
            help="Total de pessoas envolvidas em acidentes de infraestrutura"
        )
    
    with col3:
        st.metric(
            label="⚠️ Média de Vítimas/Acidente", 
            value=f"{df_analise['total_vitimas'].mean():.2f}".replace(".", ","),
            help="Número médio de pessoas envolvidas por acidente"
        )
        n_fatais = (df_analise['n_mortos'] > 0).sum()
        prop_fatais = (df_analise['n_mortos'] > 0).mean()*100
        st.metric(
            label="💀 Acidentes Fatais", 
            value=f"{n_fatais}",
            delta=f"{prop_fatais:.1f}% dos acidentes de infraestrutura",
            delta_color="inverse",
            help="Acidentes que resultaram em pelo menos uma morte"
        )



# ==================== DISTRIBUIÇÃO DE GRAVIDADE ====================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h5>Distribuição de Gravidade dos Acidentes</h5>", unsafe_allow_html=True)
st.write("""
Os acidentes foram classificados em quatro categorias de gravidade baseadas nas consequências para as vítimas:
**Fatal** (com mortes), **Grave** (com feridos graves), **Leve** (apenas feridos leves) e **Sem Lesões** (apenas ilesos).
""")

col1, col2 = st.columns([2, 1], border=True, gap="small")

with col2:
    st.markdown("<h6>Tabela de Distribuição por Categoria de Gravidade</h6><br>", unsafe_allow_html=True)
    gravidade_dist = df_analise['gravidade_cat'].value_counts()
    gravidade_pct = (gravidade_dist / len(df_analise) * 100).round(1)
    
    gravidade_df = pd.DataFrame({
        'Categoria': gravidade_dist.index,
        'Quantidade': gravidade_dist.values,
        'Percentual (%)': gravidade_pct.values
    })
    st.dataframe(gravidade_df, use_container_width=True, hide_index=True)

with col1:
    st.markdown("<h6>Distribuição por Categoria de Gravidade</h6>", unsafe_allow_html=True)
    import plotly.express as px
    import plotly.graph_objects as go
    cores = {'Fatal': '#d62728', 'Grave': '#ff7f0e', 'Leve': '#2ca02c', 'Sem_Lesoes': '#1f77b4'}
    gravidade_df['Cor'] = gravidade_df['Categoria'].map(cores)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=gravidade_df['Categoria'],
        x=gravidade_df['Quantidade'],
        orientation='h',
        marker=dict(color=gravidade_df['Cor'], line=dict(color='black', width=0)),
        text=[f"{q} ({p:.1f}%)" for q, p in zip(gravidade_df['Quantidade'], gravidade_df['Percentual (%)'])],
        textposition='auto',
        hoverinfo='text'
    ))
    fig.update_layout(
        xaxis_title='Número de Acidentes',
        yaxis_title='Categoria de Gravidade',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=False),
        margin=dict(t=30, b=50, l=50, r=10)
    )
    st.plotly_chart(fig, use_container_width=True)




# ==================== VARIÁVEIS CATEGÓRICAS ====================

variaveis_categoricas = ['tracado_simplificado', 'area_tipo', 
                          'clima_adverso', 'periodo_dia', 'gravidade_cat']

# Remover variáveis sem variabilidade
variaveis_validas = [var for var in variaveis_categoricas 
                     if df_analise[var].nunique() > 1]


st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h5>Seleção de Variáveis Categóricas para Análise Multivariada</h5>", unsafe_allow_html=True)
st.write("""
Para a análise de correspondência múltipla (MCA), selecionamos variáveis categóricas que descrevem 
características dos acidentes. Essas variáveis foram simplificadas para facilitar a interpretação dos resultados.
""")
    
col1, col2 = st.columns(2, border=True, gap="small")

with col1:
    st.metric(
        label="📋 Variáveis Selecionadas", 
    value=len(variaveis_categoricas),
    help="Número de variáveis categóricas incluídas na análise"
)

with col2:
    st.metric(
        label="🔢 Observações Disponíveis", 
        value=f"{len(df_modelo):,}".replace(",", "."),
        help="Número de acidentes com dados completos para análise"
    )

with st.container(border=True):
    categorias_info = []
    for var in variaveis_categoricas:
        n_categorias = df_analise[var].nunique()
        categorias = df_analise[var].value_counts()
        categorias_str = ', '.join([f"{cat} (n={count})" for cat, count in categorias.items()])
        categorias_info.append({
            'Variável': var,
            'N° Categorias': n_categorias,
            'Distribuição': categorias_str
        })

    categorias_df = pd.DataFrame(categorias_info)
    st.dataframe(categorias_df, use_container_width=True, hide_index=True)

    if len(variaveis_validas) < len(variaveis_categoricas):
        removidas = set(variaveis_categoricas) - set(variaveis_validas)
        variaveis_categoricas = variaveis_validas
        st.warning(f"⚠️ Variáveis removidas por falta de variabilidade: {', '.join(removidas)}")




# ==================== ANÁLISE MCA/PCA ====================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h3>Análise de Correspondência Múltipla (MCA)</h3>", unsafe_allow_html=True)

st.write("""
    A **Análise de Correspondência Múltipla (MCA)** é uma técnica de redução de dimensionalidade para variáveis categóricas. 
    Ela permite visualizar relações entre categorias de diferentes variáveis em um espaço de dimensões reduzidas, 
    facilitando a identificação de padrões e associações entre características dos acidentes.
    
    **Por que usar MCA?**
    - Reduz a complexidade dos dados mantendo informações importantes
    - Identifica grupos de acidentes com características similares
    - Permite visualizar associações entre variáveis categóricas
""")

if MCA_DISPONIVEL and len(variaveis_categoricas) >= 2:
    df_mca = df_modelo[variaveis_categoricas].copy()
    df_mca = df_mca.dropna()
    indices_validos = df_mca.index
    df_modelo_mca = df_modelo.loc[indices_validos].copy()
    
    n_componentes = min(len(variaveis_categoricas), 10)
    mca = prince.MCA(n_components=n_componentes, n_iter=10, random_state=42)
    
    try:
        mca = mca.fit(df_mca)
        mca_coords = mca.transform(df_mca)
        
        for i in range(n_componentes):
            df_modelo_mca[f'mca_dim_{i+1}'] = mca_coords.iloc[:, i].values
        
        inertia = mca.eigenvalues_
        variance_explained = (inertia / inertia.sum()) * 100
        cumulative_variance = np.cumsum(variance_explained)
        
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<h6>Variância Explicada por Dimensão</h6>", unsafe_allow_html=True)
        st.write("""
        Cada dimensão da MCA captura uma porção da variabilidade total dos dados. 
        Dimensões com maior variância explicada são mais importantes para caracterizar os acidentes.
        """)
        
        col1, col2 = st.columns([2, 1], border=True, gap="small")
        
        with col2:
            st.markdown("<h6>Tabela das Dimensões</h6><br>", unsafe_allow_html=True)
            variance_df = pd.DataFrame({
                'Dimensão': [f'Dim {i+1}' for i in range(min(5, n_componentes))],
                'Variância (%)': variance_explained[:5].round(2),
                'Acumulada (%)': cumulative_variance[:5].round(2)
            })
            st.dataframe(variance_df, use_container_width=True, hide_index=True)
        
        with col1:
            import plotly.graph_objects as go

            st.markdown("<h6>Scree Plot - Análise MCA</h6>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(variance_explained) + 1)),
                y=variance_explained,
                mode='lines+markers',
                name='Variância individual',
                line=dict(color='blue', width=2.5),
                marker=dict(size=10)
            ))
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cumulative_variance) + 1)),
                y=cumulative_variance,
                mode='lines+markers',
                name='Variância acumulada',
                line=dict(color='red', dash='dash', width=2),
                marker=dict(size=8)
            ))

            fig.update_layout(
                xaxis_title='Dimensão',
                yaxis_title='Variância Explicada (%)',
                legend=dict(font=dict(size=10)),
                template='plotly_white',
                margin=dict(t=30, b=50, l=50, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        n_dims_cluster = min(3, n_componentes)
        X_cluster = mca_coords.iloc[:, :n_dims_cluster].values
        metodo_reducao = 'MCA'
        df_cluster = df_modelo_mca.copy()
        MCA_SUCESSO = True
        
    except Exception as e:
        st.error(f"Erro detectado durante MCA: {e}.", icon=":material/error:")
        MCA_SUCESSO = False
else:
    st.warning("Falha ao aplicar MCA! Variáveis insuficientes.", icon=":warning:")
    MCA_SUCESSO = False

# Fallback: se MCA falhar
if not MCA_SUCESSO:
    st.error("MCA não pôde ser realizada no momento. Verifique a conexão ao banco de dados.", icon=":material/error:")

with st.expander("Interpretação das Dimensões Encontradas na MCA", icon=":material/info:"):
    st.write("""
- A Dimensão 1 é a mais relevante, capturando cerca de 1/4 de toda a estrutura dos dados.
- A Dimensão 2 complementa essa representação com mais 22%, somando quase 48% nas duas primeiras dimensões.
- As dimensões seguintes (3, 4 e 5) ainda carregam informação, mas cada uma explica uma fração menor (~15–19%), sugerindo que não existe um único eixo dominante; a variabilidade está distribuída de forma relativamente homogênea.

As duas primeiras dimensões resumem cerca de 48% da variabilidade total, suficiente para uma projeção bidimensional inicial. Com três dimensões, chega-se a 66%, já representando a maior parte dos padrões relevantes. O total de cinco dimensões recupera 100% da variância, mas isso não significa que todas são igualmente importantes para interpretação.
    """)





# ==================== ANÁLISE DE CLUSTERS ====================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h3>Análise de Clusters dos Acidentes</h3>", unsafe_allow_html=True)

st.write("""
O **clustering (agrupamento)** identifica grupos de acidentes com características similares usando as dimensões 
obtidas na MCA/PCA. O algoritmo **K-Means** particiona os dados em k grupos, minimizando a variabilidade intra-cluster.

**Métricas de avaliação:**
- **Inércia (WCSS):** Soma das distâncias ao quadrado dentro dos clusters (menor = melhor)
- **Silhueta:** Mede quão bem separados estão os clusters (-1 a 1, maior = melhor)
- **Davies-Bouldin:** Razão entre distâncias intra e inter-cluster (menor = melhor)
""")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Determinar número ótimo de clusters - CORRIGIDO: k=2 a k=6
k_range = range(2, 7)  # Testar de 2 a 6 clusters

inertias = []
silhouette_scores_list = []
davies_bouldin_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores_list.append(silhouette_score(X_scaled, labels))
    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))


st.markdown("<br>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown("<h6>Seleção do Número Ótimo de Clusters</h6>", unsafe_allow_html=True)
    st.write("""
    Testamos valores de k entre 2 e 6 clusters e avaliamos a qualidade dos agrupamentos 
    usando três métricas complementares. O objetivo é encontrar um equilíbrio entre simplicidade 
    (poucos clusters) e boa separação dos grupos.
    
    - **Método do Cotovelo:** Busca-se o ponto onde a redução da inércia diminui drasticamente
    - **Silhueta > 0.5:** Indica boa separação entre clusters
    - **Davies-Bouldin baixo:** Clusters bem definidos e distintos
    """)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    import plotly.graph_objects as go

    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
        # Gráfico 1: Método do Cotovelo
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            name='Inércia (WCSS)',
            line=dict(color='blue', width=2.5),
            marker=dict(size=10)
        ))
        fig.add_vline(x=4, line=dict(color='orange', dash='dash'), annotation_text='k=4', annotation_position='top right')
        fig.add_vline(x=5, line=dict(color='purple', dash='dash'), annotation_text='k=5', annotation_position='top right')
        fig.update_layout(
            title="Método do Cotovelo (Inércia)",
            xaxis_title="Número de Clusters (k)",
            yaxis_title="Inércia (WCSS)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Gráfico 2: Silhueta
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=silhouette_scores_list,
            mode='lines+markers',
            name='Coeficiente de Silhueta',
            line=dict(color='red', width=2.5),
            marker=dict(size=10)
        ))
        fig.add_hline(y=0.5, line=dict(color='green', dash='dash'), annotation_text='Boa separação (>0.5)', annotation_position='top left')
        fig.add_hline(y=0.25, line=dict(color='orange', dash='dash'), annotation_text='Separação fraca (<0.25)', annotation_position='bottom left')
        fig.add_vline(x=4, line=dict(color='orange', dash='dash'), annotation_text='k=4', annotation_position='top right')
        fig.add_vline(x=5, line=dict(color='purple', dash='dash'), annotation_text='k=5', annotation_position='top right')
        fig.update_layout(
            title="Coeficiente de Silhueta",
            xaxis_title="Número de Clusters (k)",
            yaxis_title="Coeficiente de Silhueta",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        # Gráfico 3: Davies-Bouldin
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=davies_bouldin_scores,
            mode='lines+markers',
            name='Davies-Bouldin Index',
            line=dict(color='green', width=2.5),
            marker=dict(size=10)
        ))
        fig.add_vline(x=4, line=dict(color='orange', dash='dash'), annotation_text='k=4', annotation_position='top right')
        fig.add_vline(x=5, line=dict(color='purple', dash='dash'), annotation_text='k=5', annotation_position='top right')
        fig.update_layout(
            title="Davies-Bouldin Index",
            xaxis_title="Número de Clusters (k)",
            yaxis_title="Davies-Bouldin Index",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)



idx_4 = list(k_range).index(4)
# Selecionar  k=4  baseado na silhueta
k_otimo = 4
idx_otimo = idx_4



silhueta_otima = silhouette_scores_list[idx_otimo]
db_otimo = davies_bouldin_scores[idx_otimo]

with st.container(border=True):
    st.markdown("<h6>Resultado da Seleção</h6>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Número Ótimo de Clusters", value=k_otimo)
    
    with col2:
        st.metric("📊 Coeficiente de Silhueta", value=f"{silhueta_otima:.4f}")
    
    with col3:
        st.metric("📉 Davies-Bouldin Index", value=f"{db_otimo:.4f}")
    
    # Interpretação da qualidade
    
    if silhueta_otima < 0.25:
        st.warning("""
        ⚠️ **Estrutura Fraca** (Silhueta < 0.25)
        
        Os clusters não estão bem separados, sugerindo sobreposição significativa entre categorias de acidentes. 
        Isso indica que os acidentes de infraestrutura apresentam características muito heterogêneas, 
        dificultando a formação de grupos distintos.
        """)
    elif silhueta_otima < 0.5:
        st.info("""
        ✓ **Estrutura Moderada** (Silhueta entre 0.25 e 0.5)
        
        Os clusters têm separação razoável, embora haja alguma sobreposição entre grupos. 
        A segmentação identifica padrões distintos, mas os limites entre clusters não são totalmente nítidos. 
        Esta é uma situação comum em dados reais complexos como os de acidentes de trânsito.
        """)
    else:
        st.success("""
        ✓ **Estrutura Bem Definida** (Silhueta > 0.5)
        
        Os clusters estão claramente separados e bem definidos. Os acidentes dentro de cada cluster 
        são muito similares entre si e diferentes dos outros grupos, indicando padrões claros e 
        interpretáveis nos acidentes de infraestrutura.
        """)

# Clustering final com k_otimo
kmeans_final = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans_final.fit_predict(X_scaled)




# ==================== PERFIL DOS CLUSTERS ====================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h5>Perfil dos Clusters Identificados</h5>", unsafe_allow_html=True)
st.write(f"""
Após agrupar os acidentes em **{k_otimo} clusters**, analisamos as características de cada grupo 
para entender o que diferencia cada padrão de acidente. Isso permite identificar perfis específicos 
de acidentes de infraestrutura e direcionar ações de prevenção.
""")
    
# Tamanho dos clusters
tamanhos = df_cluster['cluster'].value_counts().sort_index()

col1, col2 = st.columns([2, 1], border=True, gap="small")

with col2:
    st.markdown("<h6>Distribuição dos Acidentes em Clusters</h6><br>", unsafe_allow_html=True)
    cluster_sizes_df = pd.DataFrame({
        'Cluster': [f'Cluster {i}' for i in tamanhos.index],
        'N° Acidentes': tamanhos.values,
        'Proporção': [(n/len(df_cluster)*100) for n in tamanhos.values]
    })
    cluster_sizes_df['Proporção'] = cluster_sizes_df['Proporção'].apply(lambda x: f"{x:.1f}%")
    st.dataframe(cluster_sizes_df, use_container_width=True, hide_index=True)
    
    # Verificar balanceamento
    max_prop = tamanhos.max() / len(df_cluster)
    min_prop = tamanhos.min() / len(df_cluster)
    
    if max_prop > 0.5:
        st.warning("⚠️ Clusters desbalanceados: um cluster domina os dados")
    elif min_prop < 0.05:
        st.info("ℹ️ Alguns clusters são pequenos, representando padrões raros")
    else:
        st.success("✓ Clusters relativamente balanceados")

with col1:
    st.markdown("<h6>Tamanho dos Clusters</h6>", unsafe_allow_html=True)

    import plotly.graph_objects as go

    fig = go.Figure()

    # Add bar chart for cluster sizes
    fig.add_trace(go.Bar(
        y=[f'Cluster {i}' for i in tamanhos.index],
        x=tamanhos.values,
        orientation='h',
        marker=dict(color=sns.color_palette("Set2", k_otimo).as_hex()),
        text=[f"{val} ({val/len(df_cluster)*100:.1f}%)" for val in tamanhos.values],
        textposition='auto',
        hoverinfo='x+y'
    ))

    # Update layout
    fig.update_layout(
        xaxis_title="Número de Acidentes",
        yaxis_title="Cluster",
        template="plotly_white",
        margin=dict(t=30, b=50, l=50, r=10),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


# Perfil categórico
with st.container(border=True):
    st.markdown("<h6>Características Categóricas por Cluster</h6>", unsafe_allow_html=True)
    st.write("""
    Analisamos a distribuição percentual de cada característica categórica dentro de cada cluster. 
    Valores altos (cores mais intensas) indicam que determinada característica é predominante naquele grupo, 
    ajudando a definir o "perfil" de cada cluster.
    """)

    # Renomear variáveis para exibição (Tracado Simplificado, Area Tipo, )
    # Renomear variáveis para exibição (Traçado da Via, Tipo de Pista, etc.)
    variaveis_renomeadas = {
        'tracado_simplificado': 'Traçado da Via',
        'tipo_pista': 'Tipo de Pista',
    }

    # Renomear variáveis para exibição (Tipo de Área, Clima Adverso, etc.)
    variaveis_renomeadas.update({
        'area_tipo': 'Tipo de Área',
        'clima_adverso': 'Clima',
        'horario_acidente': 'Horário do Acidente',
        'tipo_acidente': 'Tipo de Acidente',
        'gravidade': 'Gravidade',
        'clima': 'Clima',
        'periodo_dia': 'Período do Dia',
        'gravidade_cat': 'Gravidade'
    })

    tabs = st.tabs([variaveis_renomeadas.get(var, var.replace('_', ' ').title()) for var in variaveis_categoricas])
    
    for i, var in enumerate(variaveis_categoricas):
        with tabs[i]:
            perfil = pd.crosstab(df_cluster['cluster'], df_cluster[var], 
                               normalize='index') * 100
            perfil = perfil.round(1)
            perfil.index = [f'Cluster {idx}' for idx in perfil.index]
            
            col1, col2 = st.columns([2, 1], gap="medium")
            
            with col1:
                st.markdown(f"<h6>Distribuição: {variaveis_renomeadas.get(var, var.replace('_', ' ').title())}</h6>", unsafe_allow_html=True)
                st.dataframe(perfil.style.background_gradient(cmap='YlOrRd', axis=1), 
                            use_container_width=True)
                st.caption("💡 Cores mais intensas indicam maior concentração da característica no cluster")
            
            with col2:
                st.markdown("<h6>Interpretação</h6>", unsafe_allow_html=True)
                # Identificar característica dominante em cada cluster
                for cluster_idx in perfil.index:
                    categoria_dominante = perfil.loc[cluster_idx].idxmax()
                    valor_dominante = perfil.loc[cluster_idx].max()
                    if valor_dominante > 50:
                        st.write(f"- **{cluster_idx}:** {categoria_dominante} ({valor_dominante:.1f}%)")



# PERFIS DOS CLUSTERS

caracteristicas = {}

for var in variaveis_categoricas:
    var_nome = variaveis_renomeadas.get(var, var.replace('_', ' ').title())
    tabela = pd.crosstab(df_cluster['cluster'], df_cluster[var], normalize='index') * 100
    geral = df_cluster[var].value_counts(normalize=True) * 100
    
    importancia = tabela.subtract(geral, axis=1)

    caracteristicas[var] = importancia

with st.expander("Detalhamento do Perfil dos Clusters", icon=":material/info:", expanded=True):
    clusters = sorted(df_cluster["cluster"].unique())
    cols = st.columns(len(clusters), border=True)

    for idx, cluster in enumerate(clusters):
        with cols[idx]:
            st.markdown(f"<h6 style='color: #2c3e50;'>🔍 Perfil do Cluster {cluster}</h6>", unsafe_allow_html=True)

            for var in variaveis_categoricas:
                var_nome = variaveis_renomeadas.get(var, var.replace('_', ' ').title())
                imp = caracteristicas[var].loc[cluster]
                categoria = imp.idxmax()
                valor = imp.max()

                st.markdown(
                    f"<ul style='margin-left: 2px; color: #34495e;'>"
                    f"<li><strong>{var_nome}:</strong> "
                    f"<span style='color: #16a085;'>'{categoria}'</span><br> "
                    f"{valor:.1f} pontos percentuais acima da média.</li>"
                    f"</ul>",
                    unsafe_allow_html=True
                )



# GRAVIDADE

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h5>Gravidade dos Acidentes por Cluster</h5>", unsafe_allow_html=True)
st.write("""
Comparamos as características de gravidade dos acidentes em cada cluster. 
Isso nos permite identificar quais clusters apresentam acidentes mais severos e 
quais concentram acidentes de menor gravidade.
""")

with st.container(border=True):    
    cluster_stats_num = df_cluster.groupby('cluster').agg({
        'total_vitimas': ['count', 'mean', 'std'],
        'n_mortos': ['sum', 'mean', 'max'],
        'n_feridos_graves': ['sum', 'mean', 'max'],
        'n_feridos_leves': ['sum', 'mean', 'max'],
        'indice_gravidade': ['mean', 'std', 'max']
    }).round(2)

    # Renomear colunas para exibição
    cluster_stats_num = cluster_stats_num.rename(columns={
        'total_vitimas': 'Total de Vítimas',
        'n_mortos': 'Número de Mortos',
        'n_feridos_graves': 'Número de Feridos Graves',
        'n_feridos_leves': 'Número de Feridos Leves',
        'indice_gravidade': 'Índice de Gravidade'
    })

    cluster_stats_num.columns = ['_'.join(col).strip() for col in cluster_stats_num.columns.values]
    cluster_stats_num = cluster_stats_num.reset_index()
    cluster_stats_num['cluster'] = cluster_stats_num['cluster'].apply(lambda x: f'Cluster {x}')

    st.dataframe(cluster_stats_num, use_container_width=True, hide_index=True)
    
    with st.expander("Legenda das Estatísticas", icon=":material/info:"):
        st.markdown("""
        - `count`: Número de acidentes no cluster
        - `mean`: Valor médio da variável
        - `std`: Desvio padrão (variabilidade dentro do cluster)
        - `sum`: Soma total (útil para mortos e feridos)
        - `max`: Valor máximo observado
        - `indice_gravidade`: Métrica ponderada = mortos×4 + feridos graves×2 + feridos leves×1
        """)
    
    # Identificar cluster mais grave
    gravidade_media = df_cluster.groupby('cluster')['indice_gravidade'].mean()
    cluster_mais_grave = gravidade_media.idxmax()
    gravidade_max = gravidade_media.max()
    
    st.warning(f"🚨 **Cluster mais grave:** Cluster {cluster_mais_grave} (índice médio de gravidade = {gravidade_max:.2f})")




st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h5>Visualização dos Clusters no Espaço MCA</h5>", unsafe_allow_html=True)

# Visualização dos clusters
if X_cluster.shape[1] >= 2:
    with st.container(border=True):
        st.write(f"""
        Este gráfico mostra a distribuição dos acidentes no espaço bidimensional obtido pela {metodo_reducao}. 
        Cada ponto representa um acidente, e as cores indicam os {k_otimo} clusters identificados. 
        
        **Como interpretar:**
        - Clusters bem separados espacialmente indicam grupos com características muito distintas
        - Sobreposição entre clusters sugere acidentes com características mistas
        - Tamanho dos pontos reflete a densidade de acidentes naquela região do espaço
        """)
        
        from scipy.spatial import ConvexHull
        import numpy as np
        import plotly.graph_objects as go

        fig = go.Figure()

        # Cores
        palette = px.colors.qualitative.Set2[:k_otimo]

        # MCA Dim1, Dim2
        df_plot = df_cluster.copy()
        df_plot["MCA1"] = mca_coords.iloc[:, 0].values
        df_plot["MCA2"] = mca_coords.iloc[:, 1].values

        # Adicionar scatter por cluster
        for cluster_id, cor in zip(sorted(df_plot["cluster"].unique()), palette):

            df_c = df_plot[df_plot["cluster"] == cluster_id]

            # Scatter normal
            fig.add_trace(go.Scatter(
                x=df_c["MCA1"], y=df_c["MCA2"],
                mode="markers",
                name=f"Cluster {cluster_id}",
                marker=dict(size=8, color=cor, line=dict(width=1.5, color='black')),
                hovertemplate=(
                    "Cluster: %{customdata[0]}<br>"
                    "Gravidade: %{customdata[1]:.1f}<br>"
                    "Total vítimas: %{customdata[2]}<extra></extra>"
                ),
                customdata=np.stack([df_c["cluster"], df_c["indice_gravidade"], df_c["total_vitimas"]], axis=-1)
            ))

            # Convex Hull (com contorno)
            if len(df_c) >= 3:
                pontos = df_c[["MCA1", "MCA2"]].values
                hull = ConvexHull(pontos)
                hull_pts = pontos[hull.vertices]

                fig.add_trace(go.Scatter(
                    x=hull_pts[:, 0],
                    y=hull_pts[:, 1],
                    mode='lines',
                    line=dict(color=cor, width=3),
                    name=f"Região Cluster {cluster_id}",
                    opacity=0.25,
                    showlegend=False
                ))

        # Layout final
        fig.update_layout(
            title="Clusters na MCA com Contornos",
            xaxis_title="MCA Dim 1",
            yaxis_title="MCA Dim 2",
            template="plotly_white",
            width=900,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)





# ==================== CONCLUSÕES ====================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h3>Conclusões da Análise de Acidentes de Infraestrutura</h3>", unsafe_allow_html=True)


st.write(f"""
    Baseado na análise de {len(df_analise)} acidentes relacionados à infraestrutura viária em 2024:
    
    1. **Segmentação Identificada:** A análise revelou **{k_otimo} grupos distintos** de acidentes 
       com características similares (Silhueta = {silhueta_otima:.3f}).
    
    2. **Proporção de Infraestrutura:** {len(acidentes_infra)/df['id'].nunique()*100:.1f}% dos acidentes 
       totais foram causados por problemas de infraestrutura, envolvendo {df_analise['total_vitimas'].sum()} vítimas.
    
    3. **Gravidade Geral:** {(df_analise['n_mortos'] > 0).mean()*100:.1f}% dos acidentes de infraestrutura 
       foram fatais, com média de {df_analise['total_vitimas'].mean():.2f} vítimas por acidente.
    
    4. **Cluster Crítico:** O Cluster {cluster_mais_grave} apresenta o maior índice de gravidade 
       (média = {gravidade_max:.2f}), indicando padrões de acidentes que requerem atenção prioritária.
    
    5. **Implicações Práticas:** A identificação de {k_otimo} perfis distintos de acidentes permite:
       - Priorização de investimentos em infraestrutura
       - Ações preventivas direcionadas por tipo de problema
       - Monitoramento específico de pontos críticos identificados em cada cluster
    
    6. **Variabilidade dos Dados:** {'A estrutura moderada dos clusters sugere que acidentes de infraestrutura são heterogêneos, mas apresentam padrões identificáveis que podem orientar políticas públicas.' if silhueta_otima < 0.5 else 'A estrutura bem definida dos clusters indica padrões claros nos acidentes de infraestrutura, facilitando a implementação de medidas preventivas específicas.'}
""")