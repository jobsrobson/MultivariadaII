# dashboard_acidentes.py
# =============================================================================
# DASHBOARD STREAMLIT - ANÁLISE MULTIVARIADA DE ACIDENTES DE INFRAESTRUTURA
# Base de dados: Acidentes PRF 2024 - RIDE-DF
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas para análise
try:
    import prince
    MCA_DISPONIVEL = True
except ImportError:
    MCA_DISPONIVEL = False

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.stats import kruskal, shapiro
from statsmodels.stats.multitest import multipletests

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Análise de Acidentes de Infraestrutura",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar visual
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #ff7f0e;
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# TÍTULO E DESCRIÇÃO
# =============================================================================

st.title("🚗 Análise Multivariada de Acidentes de Infraestrutura")
st.markdown("**PRF 2024 - RIDE-DF** | Análise de Correspondência Múltipla + Clusterização")

# =============================================================================
# SIDEBAR - CONFIGURAÇÕES
# =============================================================================

with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "Carregar base de dados CSV",
        type=['csv'],
        help="Arquivo CSV com dados de acidentes da PRF"
    )
    
    st.divider()
    
    # Parâmetros de análise
    st.subheader("Parâmetros da Análise")
    
    k_min = st.number_input("Clusters mínimo (k)", min_value=2, max_value=5, value=2)
    k_max = st.number_input("Clusters máximo (k)", min_value=3, max_value=15, value=10)
    
    n_dims_mca = st.slider("Dimensões MCA para cluster", min_value=2, max_value=5, value=3)
    
    st.divider()
    
    # Filtros
    st.subheader("Filtros")
    filtrar_outliers = st.checkbox("Remover outliers (P99)", value=True)
    
    st.divider()
    
    st.info("💡 **Dica:** Os dados são processados em tempo real. Ajuste os parâmetros para explorar diferentes configurações.")

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

@st.cache_data
def carregar_dados(file):
    """Carrega dados do CSV"""
    if file is not None:
        df = pd.read_csv(file, sep=';', low_memory=False)
    else:
        # Dados de exemplo/teste
        st.warning("⚠️ Usando dados de exemplo. Faça upload do arquivo CSV na barra lateral.")
        return None
    return df

def agregar_vitimas(group):
    """Agregação corrigida de vítimas (pessoas únicas)"""
    # Remover duplicatas de pessoas
    group_unico = group.drop_duplicates(subset='pesid', keep='first')
    
    # Contagem por estado físico
    estados = group_unico['estado_fisico'].value_counts().to_dict()
    
    # Calcular totais
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

def simplificar_tracado(x):
    """Simplifica categorias de traçado"""
    x_str = str(x).lower()
    if 'curva' in x_str:
        return 'Curva'
    elif 'aclive' in x_str or 'declive' in x_str:
        return 'Inclinacao'
    elif 'ponte' in x_str or 'viaduto' in x_str:
        return 'Ponte_Viaduto'
    else:
        return 'Reta'

def categorizar_gravidade(row):
    """Categoriza gravidade do acidente"""
    if row['n_mortos'] > 0:
        return 'Fatal'
    elif row['n_feridos_graves'] > 0:
        return 'Grave'
    elif row['n_feridos_leves'] > 0:
        return 'Leve'
    else:
        return 'Sem_Lesoes'

# =============================================================================
# PROCESSAMENTO DOS DADOS
# =============================================================================

# Carregar dados
df_raw = carregar_dados(uploaded_file)

if df_raw is not None:
    
    # Causas de infraestrutura
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
    
    with st.spinner('Processando dados...'):
        
        # Filtrar acidentes de infraestrutura
        df_raw['causa_infraestrutura'] = df_raw['causa_acidente'].isin(causas_infraestrutura)
        acidentes_infra = df_raw[df_raw['causa_infraestrutura']]['id'].unique()
        df_infra = df_raw[df_raw['id'].isin(acidentes_infra)].copy()
        
        # Agregar vítimas por acidente
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
        
        # Juntar
        df_analise = df_acidentes.merge(df_vitimas, on='id', how='inner')
        
        # Variáveis derivadas
        df_analise['indice_gravidade'] = (
            df_analise['n_mortos'] * 4 +
            df_analise['n_feridos_graves'] * 2 +
            df_analise['n_feridos_leves'] * 1
        )
        
        df_analise['gravidade_cat'] = df_analise.apply(categorizar_gravidade, axis=1)
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
        
        # Remover outliers se solicitado
        df_modelo = df_analise.copy()
        
        if filtrar_outliers:
            for col in ['total_vitimas', 'n_mortos', 'n_feridos_graves', 'indice_gravidade']:
                q99 = df_modelo[col].quantile(0.99)
                df_modelo[col] = df_modelo[col].clip(upper=q99)
        
        # Variáveis para MCA
        variaveis_categoricas = ['tracado_simplificado', 'area_tipo', 
                                  'clima_adverso', 'periodo_dia', 'gravidade_cat']
        
        # Remover variáveis sem variabilidade
        variaveis_validas = [var for var in variaveis_categoricas 
                             if df_modelo[var].nunique() > 1]
        
    # =============================================================================
    # SEÇÃO 1: ESTATÍSTICAS GERAIS
    # =============================================================================
    
    st.header("📊 Visão Geral dos Dados")
    
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total de Acidentes",
                f"{len(df_analise):,}",
                help="Acidentes de infraestrutura identificados"
            )
        
        with col2:
            st.metric(
                "Média Vítimas/Acidente",
                f"{df_analise['total_vitimas'].mean():.2f}",
                help="Pessoas únicas envolvidas por acidente"
            )
        
        with col3:
            st.metric(
                "Acidentes Fatais",
                f"{(df_analise['n_mortos'] > 0).sum()}",
                f"{(df_analise['n_mortos'] > 0).mean()*100:.1f}%"
            )
        
        with col4:
            st.metric(
                "Total de Mortes",
                f"{df_analise['n_mortos'].sum():.0f}",
                help="Total de óbitos registrados"
            )
    
    # Distribuição de gravidade
    with st.container(border=True):
        st.subheader("Distribuição por Gravidade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de pizza
            grav_counts = df_analise['gravidade_cat'].value_counts()
            fig_pizza = px.pie(
                values=grav_counts.values,
                names=grav_counts.index,
                title="Proporção por Categoria de Gravidade",
                color_discrete_sequence=px.colors.sequential.Reds_r
            )
            st.plotly_chart(fig_pizza, use_container_width=True)
        
        with col2:
            # Histograma do índice de gravidade
            fig_hist = px.histogram(
                df_analise,
                x='indice_gravidade',
                nbins=30,
                title="Distribuição do Índice de Gravidade",
                labels={'indice_gravidade': 'Índice de Gravidade'},
                color_discrete_sequence=['#ff7f0e']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # =============================================================================
    # SEÇÃO 2: MCA - REDUÇÃO DE DIMENSIONALIDADE
    # =============================================================================
    
    st.header("🔍 Análise de Correspondência Múltipla (MCA)")
    
    with st.spinner('Executando MCA...'):
        
        MCA_SUCESSO = False
        
        if MCA_DISPONIVEL and len(variaveis_validas) >= 2:
            df_mca = df_modelo[variaveis_validas].dropna()
            indices_validos = df_mca.index
            df_modelo_mca = df_modelo.loc[indices_validos].copy()
            
            n_componentes = min(len(variaveis_validas), 10)
            mca = prince.MCA(n_components=n_componentes, n_iter=10, random_state=42)
            
            try:
                mca = mca.fit(df_mca)
                mca_coords = mca.transform(df_mca)
                
                for i in range(n_componentes):
                    df_modelo_mca[f'mca_dim_{i+1}'] = mca_coords.iloc[:, i].values
                
                variancia_expl = mca.eigenvalues_ / mca.eigenvalues_.sum()
                
                X_cluster = mca_coords.iloc[:, :n_dims_mca].values
                metodo_reducao = 'MCA'
                df_cluster = df_modelo_mca.copy()
                
                MCA_SUCESSO = True
                
            except:
                MCA_SUCESSO = False
        
        # Fallback PCA
        if not MCA_SUCESSO:
            st.warning("⚠️ Usando PCA como alternativa à MCA")
            
            df_dummy = pd.get_dummies(df_modelo[variaveis_validas], drop_first=True)
            variancia = df_dummy.var()
            colunas_validas = variancia[variancia > 0].index.tolist()
            df_dummy = df_dummy[colunas_validas]
            
            n_componentes_pca = min(5, len(colunas_validas))
            pca = PCA(n_components=n_componentes_pca, random_state=42)
            X_cluster = pca.fit_transform(df_dummy)
            
            variancia_expl = pca.explained_variance_ratio_
            
            for i in range(n_componentes_pca):
                df_modelo[f'pca_comp_{i+1}'] = X_cluster[:, i]
            
            metodo_reducao = 'PCA'
            df_cluster = df_modelo.copy()
    
    with st.container(border=True):
        st.subheader(f"Variância Explicada - {metodo_reducao}")
        
        # Scree plot
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Scatter(
            x=list(range(1, len(variancia_expl) + 1)),
            y=variancia_expl * 100,
            mode='lines+markers',
            marker=dict(size=10, color='#1f77b4'),
            line=dict(width=2, color='#1f77b4')
        ))
        
        fig_scree.update_layout(
            title="Scree Plot - Variância Explicada por Dimensão",
            xaxis_title="Dimensão",
            yaxis_title="Variância Explicada (%)",
            height=400
        )
        
        st.plotly_chart(fig_scree, use_container_width=True)
        
        # Tabela de variância
        col1, col2 = st.columns(2)
        
        with col1:
            var_df = pd.DataFrame({
                'Dimensão': range(1, min(6, len(variancia_expl)+1)),
                'Variância (%)': [f"{v*100:.2f}%" for v in variancia_expl[:5]]
            })
            st.dataframe(var_df, use_container_width=True, hide_index=True)
        
        with col2:
            var_acum = np.cumsum(variancia_expl[:5])
            var_acum_df = pd.DataFrame({
                'Dimensões': [f"1-{i+1}" for i in range(min(5, len(var_acum)))],
                'Variância Acumulada (%)': [f"{v*100:.2f}%" for v in var_acum]
            })
            st.dataframe(var_acum_df, use_container_width=True, hide_index=True)
    
    # =============================================================================
    # SEÇÃO 3: CLUSTERIZAÇÃO
    # =============================================================================
    
    st.header("🎯 Análise de Clusters")
    
    with st.spinner('Determinando clusters ótimos...'):
        
        # Padronizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # Determinar k ótimo
        k_range = range(k_min, min(k_max + 1, len(df_cluster) // 10))
        
        inertias = []
        silhouette_scores_list = []
        davies_bouldin_scores_list = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores_list.append(silhouette_score(X_scaled, labels))
            davies_bouldin_scores_list.append(davies_bouldin_score(X_scaled, labels))
        
        k_otimo = list(k_range)[np.argmax(silhouette_scores_list)]
        silhueta_otima = max(silhouette_scores_list)
        db_otimo = davies_bouldin_scores_list[np.argmax(silhouette_scores_list)]
        
        # Clustering final
        kmeans_final = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
        df_cluster['cluster'] = kmeans_final.fit_predict(X_scaled)
    
    with st.container(border=True):
        st.subheader("Seleção do Número Ótimo de Clusters")
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Clusters Ótimo (k)", k_otimo)
        
        with col2:
            st.metric("Silhueta", f"{silhueta_otima:.3f}")
        
        with col3:
            st.metric("Davies-Bouldin", f"{db_otimo:.3f}")
        
        # Gráficos de métricas
        fig_metricas = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Método do Cotovelo", "Coeficiente de Silhueta", "Davies-Bouldin Index")
        )
        
        fig_metricas.add_trace(
            go.Scatter(x=list(k_range), y=inertias, mode='lines+markers', name='Inércia'),
            row=1, col=1
        )
        
        fig_metricas.add_trace(
            go.Scatter(x=list(k_range), y=silhouette_scores_list, mode='lines+markers', 
                      name='Silhueta', marker=dict(color='red')),
            row=1, col=2
        )
        
        fig_metricas.add_trace(
            go.Scatter(x=list(k_range), y=davies_bouldin_scores_list, mode='lines+markers', 
                      name='DB Index', marker=dict(color='green')),
            row=1, col=3
        )
        
        fig_metricas.update_xaxes(title_text="Número de Clusters (k)")
        fig_metricas.update_layout(height=400, showlegend=False)
        
        st.plotly_chart(fig_metricas, use_container_width=True)
    
    # Visualização dos clusters
    with st.container(border=True):
        st.subheader("Visualização dos Clusters no Espaço MCA/PCA")
        
        if X_cluster.shape[1] >= 2:
            fig_clusters = px.scatter(
                x=X_cluster[:, 0],
                y=X_cluster[:, 1],
                color=df_cluster['cluster'].astype(str),
                title=f"Clusters (k={k_otimo}) - {metodo_reducao}",
                labels={'x': f'{metodo_reducao} - Dimensão 1', 
                       'y': f'{metodo_reducao} - Dimensão 2',
                       'color': 'Cluster'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            fig_clusters.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
            fig_clusters.update_layout(height=600)
            
            st.plotly_chart(fig_clusters, use_container_width=True)
    
    # =============================================================================
    # SEÇÃO 4: PERFIL DOS CLUSTERS
    # =============================================================================
    
    st.header("📈 Perfil dos Clusters")
    
    # Estatísticas por cluster
    cluster_stats = df_cluster.groupby('cluster').agg({
        'total_vitimas': ['count', 'mean'],
        'n_mortos': ['sum', 'mean'],
        'n_feridos_graves': ['sum', 'mean'],
        'n_feridos_leves': ['sum', 'mean'],
        'indice_gravidade': ['mean', 'std']
    }).round(2)
    
    with st.container(border=True):
        st.subheader("Estatísticas Numéricas por Cluster")
        
        # Reformatar tabela
        cluster_stats_display = cluster_stats.copy()
        cluster_stats_display.columns = [' '.join(col).strip() for col in cluster_stats_display.columns]
        
        st.dataframe(cluster_stats_display, use_container_width=True)
    
    # Boxplots
    with st.container(border=True):
        st.subheader("Distribuição de Variáveis por Cluster")
        
        var_selecionada = st.selectbox(
            "Selecione a variável:",
            options=['total_vitimas', 'n_mortos', 'n_feridos_graves', 
                    'n_feridos_leves', 'indice_gravidade'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        fig_box = px.box(
            df_cluster,
            x='cluster',
            y=var_selecionada,
            title=f"Distribuição de {var_selecionada.replace('_', ' ').title()} por Cluster",
            color='cluster',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig_box.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Perfil categórico
    with st.container(border=True):
        st.subheader("Perfil Categórico dos Clusters")
        
        var_cat_selecionada = st.selectbox(
            "Selecione a variável categórica:",
            options=variaveis_validas,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        perfil_cat = pd.crosstab(df_cluster['cluster'], 
                                  df_cluster[var_cat_selecionada], 
                                  normalize='index') * 100
        
        fig_heatmap = px.imshow(
            perfil_cat,
            labels=dict(x="Categoria", y="Cluster", color="Proporção (%)"),
            title=f"Distribuição de {var_cat_selecionada.replace('_', ' ').title()} por Cluster",
            color_continuous_scale='YlOrRd',
            text_auto='.1f'
        )
        
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # =============================================================================
    # SEÇÃO 5: TESTES ESTATÍSTICOS
    # =============================================================================
    
    st.header("📊 Testes Estatísticos")
    
    with st.spinner('Executando testes...'):
        
        variaveis_dep = {
            'total_vitimas': 'Total de Vítimas',
            'n_mortos': 'Número de Mortos',
            'n_feridos_graves': 'Feridos Graves',
            'n_feridos_leves': 'Feridos Leves',
            'indice_gravidade': 'Índice de Gravidade'
        }
        
        resultados = []
        
        for var, nome in variaveis_dep.items():
            grupos = [df_cluster[df_cluster['cluster'] == c][var].dropna().values
                      for c in sorted(df_cluster['cluster'].unique())]
            
            h_stat, p_value = kruskal(*grupos)
            
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            
            resultados.append({
                'Variável': nome,
                'Estatística': f"{h_stat:.3f}",
                'p-valor': f"{p_value:.4f}",
                'Significância': sig
            })
        
        # Correção de Bonferroni
        p_values = [float(r['p-valor']) for r in resultados]
        reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
        
        for i, r in enumerate(resultados):
            r['p-ajustado (Bonferroni)'] = f"{p_corrected[i]:.4f}"
            r['Significativo (Bonferroni)'] = 'Sim' if reject[i] else 'Não'
    
    with st.container(border=True):
        st.subheader("Resultados dos Testes de Kruskal-Wallis")
        
        df_resultados = pd.DataFrame(resultados)
        
        st.dataframe(
            df_resultados,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Significância": st.column_config.TextColumn(
                    "Significância",
                    help="*** p<0.001, ** p<0.01, * p<0.05, ns não significativo"
                )
            }
        )
        
        st.caption("Legenda: *** p<0.001, ** p<0.01, * p<0.05, ns não significativo")
    
    # =============================================================================
    # SEÇÃO 6: EXPORTAÇÃO
    # =============================================================================
    
    st.header("💾 Exportar Resultados")
    
    with st.container(border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Download dados com clusters
            csv_clusters = df_cluster.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Dados com Clusters (CSV)",
                data=csv_clusters,
                file_name='acidentes_com_clusters.csv',
                mime='text/csv'
            )
        
        with col2:
            # Download estatísticas
            csv_stats = cluster_stats.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Estatísticas dos Clusters (CSV)",
                data=csv_stats,
                file_name='estatisticas_clusters.csv',
                mime='text/csv'
            )
    

