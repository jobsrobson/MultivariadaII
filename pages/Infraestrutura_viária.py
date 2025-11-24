import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/toledo-cia/Documents/Projetos/MultivariadaII')

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
    print("⚠️ Instale: pip install prince")
    MCA_DISPONIVEL = False

# Bibliotecas para análise
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.stats import f_oneway, kruskal, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# Configuração visual
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Funções

def agregar_vitimas(group):
    """Conta vítimas por estado físico em cada acidente"""
    estados = group['estado_fisico'].value_counts().to_dict()
    
    return pd.Series({
        'total_vitimas': len(group),
        'n_mortos': estados.get('Óbito', 0) + estados.get('Morte', 0),
        'n_feridos_graves': estados.get('Lesões Graves', 0),
        'n_feridos_leves': estados.get('Lesões Leves', 0),
        'n_ilesos': estados.get('Ileso', 0),
        'n_ignorados': len(group) - sum([
            estados.get('Óbito', 0),
            estados.get('Morte', 0),
            estados.get('Lesões Graves', 0),
            estados.get('Lesões Leves', 0),
            estados.get('Ileso', 0)
        ])
    })

# Categorizar gravidade
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

# Simplificar variáveis categóricas
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

st.title("Análise da Hipótese 3: Infraestrutura Viária como Causa de Acidentes mais Graves")

st.write("""
    Nesta análise, investigamos se determinadas causas relacionadas à infraestrutura viária estão associadas a acidentes de trânsito mais graves na RIDE-DF. Utilizamos dados agregados de acidentes e aplicamos testes estatísticos para identificar quais causas de infraestrutura estão correlacionadas com índices de gravidade mais elevados.\n\n

    Os dados utilizados nesta análise foram extraídos diretamente do site da **Polícia Rodoviária Federal (PRF)**, são os dados agregados por pessoas e acidentes de **2024**. 
""")
df = fetch_data("SELECT * FROM acidente_transito")

# Preparação dos dados
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

df_vitimas = df_infra.groupby('id').apply(agregar_vitimas).reset_index()

caracteristicas_acidente = [
    'data_inversa', 'dia_semana', 'horario', 'uf', 'br', 'km',
    'municipio', 'causa_principal', 'causa_acidente', 'tipo_acidente',
    'classificacao_acidente', 'fase_dia', 'sentido_via',
    'condicao_metereologica', 'tipo_pista', 'tracado_via', 'uso_solo',
    'latitude', 'longitude'
]

colunas_disponiveis = [col for col in caracteristicas_acidente if col in df_infra.columns]

# CORREÇÃO: Usar drop_duplicates ao invés de groupby para evitar erro
df_acidentes = df_infra[['id'] + colunas_disponiveis].drop_duplicates(subset='id', keep='first')

# Juntar contagens de vítimas
df_analise = df_acidentes.merge(df_vitimas, on='id', how='inner')

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

# Criar identificador de município
df_analise['municipio_clean'] = df_analise['municipio'].fillna('Desconhecido').astype(str)

variaveis_categoricas = ['tracado_simplificado', 'area_tipo', 
                          'clima_adverso', 'periodo_dia', 'gravidade_cat']

# Remover variáveis sem variabilidade
variaveis_validas = [var for var in variaveis_categoricas 
                     if df_analise[var].nunique() > 1]

# Dataset para análise
df_modelo = df_analise.copy()

# Remover outliers extremos (percentil 99)
for col in ['total_vitimas', 'n_mortos', 'n_feridos_graves', 'indice_gravidade']:
    q99 = df_modelo[col].quantile(0.99)
    df_modelo[col] = df_modelo[col].clip(upper=q99)


# =========================  Dashboard  =========================
st.header("Resumo dos Dados de Acidentes e Vítimas e estatísticas Descritivas", divider=True)


col1, col2, col3 = st.columns(3)

with col1:
    st.write(f"Total de registros (vítimas): **{len(df):,}**")
    st.write(f"Acidentes únicos: **{df['id'].nunique():,}**")
    st.write(f"Média de vítimas por acidente: **{len(df) / df['id'].nunique():.2f}**")

with col2:
    st.write(f"\nAcidentes de infraestrutura: **{len(acidentes_infra):,} ({len(acidentes_infra)/df['id'].nunique()*100:.1f}%)**")
    st.write(f"Vítimas desses acidentes: **{len(df_infra):,} ({len(df_infra)/len(df)*100:.1f}%)**")
    st.write(f"Acidentes únicos para análise: **{len(df_analise):,}**")

with col3:
    st.write(f"Média de vítimas por acidente: **{df_analise['total_vitimas'].mean():.2f}**")
    st.write(f"Acidentes fatais: **{(df_analise['n_mortos'] > 0).sum()} ({(df_analise['n_mortos'] > 0).mean()*100:.1f}%)**")
    st.write(f"Municípios únicos: **{df_analise['municipio_clean'].nunique()}**")

st.subheader("\nVariabilidade das variáveis categóricas:", divider=True)
for var in variaveis_categoricas:
    n_categorias = df_analise[var].nunique()
    categorias = df_analise[var].value_counts().to_dict()
    st.write(f"  {var}: {n_categorias} categorias - {categorias}")

if len(variaveis_validas) < len(variaveis_categoricas):
    removidas = set(variaveis_categoricas) - set(variaveis_validas)
    variaveis_categoricas = variaveis_validas
    st.write(f"\n⚠️ Variáveis removidas (sem variabilidade): {removidas}")

st.write(f"\nVariáveis para análise: {len(variaveis_categoricas)}")

st.write(f"Observações para análise: {len(df_modelo)}")

# ==============================  Analise multivariada  ==============================
# ====================  ANÁLISE DE CORRESPONDÊNCIA MÚLTIPLA (MCA) ====================

st.header("Análise de Correspondência Múltipla (MCA) das Causas de Infraestrutura", divider=True)

if MCA_DISPONIVEL and len(variaveis_categoricas) >= 2:
    df_mca = df_modelo[variaveis_categoricas].copy()
    
    # Remover linhas com valores faltantes
    df_mca = df_mca.dropna()
    indices_validos = df_mca.index
    df_modelo_mca = df_modelo.loc[indices_validos].copy()
    
    st.write(f"\nObservações após limpeza: {len(df_mca)}")
    
    # Configurar MCA
    n_componentes = min(len(variaveis_categoricas), 10)
    mca = prince.MCA(n_components=n_componentes, n_iter=10, random_state=42)
    
    try:
        mca = mca.fit(df_mca)
        mca_coords = mca.transform(df_mca)
        
        # Adicionar coordenadas ao dataframe
        for i in range(n_componentes):
            df_modelo_mca[f'mca_dim_{i+1}'] = mca_coords.iloc[:, i].values
        
        # Variância explicada
        inertia = mca.eigenvalues_
        variance_explained = (inertia / inertia.sum()) * 100
        cumulative_variance = np.cumsum(variance_explained)
        
        st.subheader(f"\nVariância explicada:")
        for i in range(min(5, n_componentes)):
            st.write(f"  Dimensão {i+1}: {variance_explained[i]:.2f}% (acumulada: {cumulative_variance[i]:.2f}%)")
        
        # Scree plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(variance_explained) + 1), variance_explained, 
                'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Dimensão', fontsize=12)
        ax.set_ylabel('Variância Explicada (%)', fontsize=12)
        ax.set_title('Scree Plot - MCA', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Usar primeiras dimensões para cluster
        n_dims_cluster = min(3, n_componentes)
        X_cluster = mca_coords.iloc[:, :n_dims_cluster].values
        metodo_reducao = 'MCA'
        df_cluster = df_modelo_mca.copy()
        
        MCA_SUCESSO = True
        
    except Exception as e:
        print(f"\n⚠️ Erro na MCA: {e}")
        print("Usando PCA como alternativa...")
        MCA_SUCESSO = False
        
else:
    st.write("\nMCA não disponível ou variáveis insuficientes. Usando PCA...")
    MCA_SUCESSO = False

# Fallback: PCA se MCA falhar
if not MCA_SUCESSO:
    df_dummy = pd.get_dummies(df_modelo[variaveis_categoricas], drop_first=True)
    
    # Remover colunas com variância zero
    variancia = df_dummy.var()
    colunas_validas = variancia[variancia > 0].index.tolist()
    df_dummy = df_dummy[colunas_validas]
    
    st.write(f"\nVariáveis dummy criadas: {len(colunas_validas)}")
    
    n_componentes_pca = min(5, len(colunas_validas))
    pca = PCA(n_components=n_componentes_pca, random_state=42)
    X_cluster = pca.fit_transform(df_dummy)
    
    variance_explained = pca.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(variance_explained)
    
    st.write(f"\nVariância explicada (PCA):")
    for i in range(n_componentes_pca):
        st.write(f"  PC{i+1}: {variance_explained[i]:.2f}% (acumulada: {cumulative_variance[i]:.2f}%)")
    
    # Scree plot PCA
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, n_componentes_pca + 1), variance_explained, 
            'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Componente Principal', fontsize=12)
    ax.set_ylabel('Variância Explicada (%)', fontsize=12)
    ax.set_title('Scree Plot - PCA', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    for i in range(n_componentes_pca):
        df_modelo[f'pca_comp_{i+1}'] = X_cluster[:, i]
    
    metodo_reducao = 'PCA'
    df_cluster = df_modelo.copy()


st.header("Análise de Clusters dos Acidentes com Base nas Dimensões Reduzidas", divider=True)

st.write("\nDeterminando número ótimo de clusters...")
