# =============================================================================
# ANÁLISE MULTIVARIADA - INFRAESTRUTURA VIÁRIA (VERSÃO FINAL OTIMIZADA)
# Base de dados: Acidentes PRF 2024 - RIDE-DF
# Estrutura: Vítima-por-linha → Acidente → Análise
# Versão 3.0
# =============================================================================

"""

https://www.perplexity.ai/search/tenho-o-seguinte-trabalho-de-a-QzyhG49UQiWw04q_CcV8UA#19

Técnicas Estatísticas Utilizadas

Agregação de Dados:
Cada linha do dataset representa uma vítima. Os dados foram agregados por acidente (ID do acidente), permitindo calcular corretamente o total de vítimas, mortos, feridos graves, feridos leves e ilesos por ocorrência.

Construção de Variáveis:
- Índice de gravidade: soma ponderada (4x mortos, 2x feridos graves, 1x feridos leves).
- Classificação categórica da gravidade do acidente (Leve/Grave/Fatal/Sem Lesões).
- Simplificações em variáveis viárias e ambientais (tracado da via, zona urbana/rural, clima, período do dia).

Análise de Correspondência Múltipla (MCA):
    Aplicada nas variáveis categóricas relevantes para identificar padrões latentes. As 5 primeiras dimensões do MCA explicaram 100% da variância, com as 3 primeiras já explicando cerca de 67% da variação nos perfis de acidente.

Clusterização (K-Means):
    Utilizou as dimensões reduzidas do MCA. O número ótimo foi k = 10, determinado pelo coeficiente de silhueta (0.373, estrutura moderada) e pelo índice Davies-Bouldin (0.93, bom).

Testes Estatísticos (Kruskal-Wallis e Bonferroni):
    Houve comparação multigrupos para:
    - Total de vítimas
    - Número de mortos
    - Feridos graves
    - Feridos leves

Índice de gravidade
    Após correção para múltiplas comparações, 4 dessas variáveis se mantiveram estatisticamente significantes (nível ajustado α=0,01): total de vítimas, mortos, feridos graves e índice de gravidade.

Testes Post-Hoc (Tukey HSD):
    Identificaram em quais pares de clusters estavam as diferenças mais marcantes, sobretudo destacando o Cluster 4 como significativamente mais grave e letal.
"""

"""
Principais Resultados

Gravidade Discrepante:
- Acidentes de infraestrutura representam apenas 11% dos eventos, mas concentram 21,9% das vítimas — acidentes duas vezes mais graves que a média do banco.

Clusters Relevantes:
- Foram identificados 10 grupos distintos de acidentes, cada qual com perfil viário, ambiental e de desfecho diferente.​
- O Cluster 4 se destacou: 70% dos acidentes desse grupo são fatais, com média de 8 mortos por acidente, quase exclusivamente à noite, em áreas rurais e trechos de inclinação.

Diferenças Estatísticas:
- Houve diferenças estatisticamente significativas (p < 0,01 após Bonferroni) entre clusters para mortalidade, feridos graves, total de vítimas e índice de gravidade.
- Comparações post-hoc mostram que o Cluster 4 é muito mais letal e crítico do que qualquer outro, sustentando a priorização desses trechos para intervenções.

Variáveis Explicativas:
As variáveis que mais explicam a formação dos clusters e a gravidade foram:

- Geometria da via (reta, inclinação, curva, obras de arte* (pontes/viadutos))
- Área (urbana/rural)
- Período do dia (noite/dia/transição)
- Clima
- Classificação de gravidade

* "Obra de Arte" é um termo usado em Engenharia Viária para descrever trechos com construções especiais como pontes, viadutos, túneis e passarelas.

Conclusão
A análise multivariada robusta revelou que acidentes associados a infraestrutura rodoviária possuem maior gravidade, destacando um grupo de trechos críticos (rural, inclinação, noite) como focos prioritários para políticas de segurança viária e infraestrutura, com evidência estatística sólida para embasar recomendações de intervenção.

"""



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

print("="*80)
print("ANÁLISE MULTIVARIADA - ACIDENTES DE INFRAESTRUTURA (VERSÃO OTIMIZADA)")
print("="*80)

# =============================================================================
# 1. CARREGAMENTO E AGREGAÇÃO POR ACIDENTE
# =============================================================================

print("\n[1/8] Carregando e processando dados...")

# Carregar dados
df = pd.read_csv('/home/jobsr/Documents/MultivariadaII/data/acidente_transito.csv', sep=';', low_memory=False)

print(f"Total de registros (vítimas): {len(df):,}")
print(f"Acidentes únicos: {df['id'].nunique():,}")
print(f"Média de vítimas por acidente: {len(df) / df['id'].nunique():.2f}")

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

# Filtrar acidentes de infraestrutura
df['causa_infraestrutura'] = df['causa_acidente'].isin(causas_infraestrutura)
acidentes_infra = df[df['causa_infraestrutura']]['id'].unique()
df_infra = df[df['id'].isin(acidentes_infra)].copy()

print(f"\nAcidentes de infraestrutura: {len(acidentes_infra):,} ({len(acidentes_infra)/df['id'].nunique()*100:.1f}%)")
print(f"Vítimas desses acidentes: {len(df_infra):,} ({len(df_infra)/len(df)*100:.1f}%)")

# =============================================================================
# 2. AGREGAR VÍTIMAS POR ACIDENTE
# =============================================================================

print("\n[2/8] Agregando vítimas por acidente...")

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

df_vitimas = df_infra.groupby('id').apply(agregar_vitimas).reset_index()

# Características do acidente (usar primeira vítima de cada acidente)
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

print(f"Acidentes únicos para análise: {len(df_analise):,}")
print(f"Média de vítimas por acidente: {df_analise['total_vitimas'].mean():.2f}")
print(f"Acidentes fatais: {(df_analise['n_mortos'] > 0).sum()} ({(df_analise['n_mortos'] > 0).mean()*100:.1f}%)")

# =============================================================================
# 3. CRIAR VARIÁVEIS DERIVADAS
# =============================================================================

print("\n[3/8] Criando variáveis derivadas...")

# Índice de gravidade
df_analise['indice_gravidade'] = (
    df_analise['n_mortos'] * 4 +
    df_analise['n_feridos_graves'] * 2 +
    df_analise['n_feridos_leves'] * 1
)

# Categorizar gravidade
def categorizar_gravidade(row):
    if row['n_mortos'] > 0:
        return 'Fatal'
    elif row['n_feridos_graves'] > 0:
        return 'Grave'
    elif row['n_feridos_leves'] > 0:
        return 'Leve'
    else:
        return 'Sem_Lesoes'

df_analise['gravidade_cat'] = df_analise.apply(categorizar_gravidade, axis=1)

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

print(f"Municípios únicos: {df_analise['municipio_clean'].nunique()}")

# =============================================================================
# 4. ANÁLISE NO NÍVEL DE ACIDENTE (SEM AGREGAÇÃO ADICIONAL)
# =============================================================================

print("\n[4/8] Preparando dados para análise multivariada...")

# DECISÃO: Análise diretamente no nível de ACIDENTE (N=236)
# Razão: Agregação por segmento reduz N para 22, insuficiente para testes robustos

# Selecionar variáveis categóricas para MCA
variaveis_categoricas = ['tracado_simplificado', 'area_tipo', 
                          'clima_adverso', 'periodo_dia', 'gravidade_cat']

# IMPORTANTE: Remover 'pista_tipo' pois tem apenas uma categoria (100% Simples)
# Verificar variabilidade
print("\nVariabilidade das variáveis categóricas:")
for var in variaveis_categoricas:
    n_categorias = df_analise[var].nunique()
    categorias = df_analise[var].value_counts().to_dict()
    print(f"  {var}: {n_categorias} categorias - {categorias}")
    
# Remover variáveis sem variabilidade
variaveis_validas = [var for var in variaveis_categoricas 
                     if df_analise[var].nunique() > 1]

if len(variaveis_validas) < len(variaveis_categoricas):
    removidas = set(variaveis_categoricas) - set(variaveis_validas)
    print(f"\n⚠️ Variáveis removidas (sem variabilidade): {removidas}")
    variaveis_categoricas = variaveis_validas

print(f"\nVariáveis para análise: {len(variaveis_categoricas)}")

# Dataset para análise
df_modelo = df_analise.copy()

# Remover outliers extremos (percentil 99)
for col in ['total_vitimas', 'n_mortos', 'n_feridos_graves', 'indice_gravidade']:
    q99 = df_modelo[col].quantile(0.99)
    df_modelo[col] = df_modelo[col].clip(upper=q99)

print(f"Observações para análise: {len(df_modelo)}")

# =============================================================================
# 5. ANÁLISE DE CORRESPONDÊNCIA MÚLTIPLA (MCA)
# =============================================================================

print("\n" + "="*80)
print("[5/8] ANÁLISE DE CORRESPONDÊNCIA MÚLTIPLA (MCA)")
print("="*80)

if MCA_DISPONIVEL and len(variaveis_categoricas) >= 2:
    df_mca = df_modelo[variaveis_categoricas].copy()
    
    # Remover linhas com valores faltantes
    df_mca = df_mca.dropna()
    indices_validos = df_mca.index
    df_modelo_mca = df_modelo.loc[indices_validos].copy()
    
    print(f"\nObservações após limpeza: {len(df_mca)}")
    
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
        
        print(f"\nVariância explicada:")
        for i in range(min(5, n_componentes)):
            print(f"  Dimensão {i+1}: {variance_explained[i]:.2f}% (acumulada: {cumulative_variance[i]:.2f}%)")
        
        # Scree plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(variance_explained) + 1), variance_explained, 
                'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Dimensão', fontsize=12)
        ax.set_ylabel('Variância Explicada (%)', fontsize=12)
        ax.set_title('Scree Plot - MCA', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('1_mca_scree_plot.png', bbox_inches='tight')
        print("\n✓ Gráfico salvo: 1_mca_scree_plot.png")
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
    print("\nMCA não disponível ou variáveis insuficientes. Usando PCA...")
    MCA_SUCESSO = False

# Fallback: PCA se MCA falhar
if not MCA_SUCESSO:
    df_dummy = pd.get_dummies(df_modelo[variaveis_categoricas], drop_first=True)
    
    # Remover colunas com variância zero
    variancia = df_dummy.var()
    colunas_validas = variancia[variancia > 0].index.tolist()
    df_dummy = df_dummy[colunas_validas]
    
    print(f"\nVariáveis dummy criadas: {len(colunas_validas)}")
    
    n_componentes_pca = min(5, len(colunas_validas))
    pca = PCA(n_components=n_componentes_pca, random_state=42)
    X_cluster = pca.fit_transform(df_dummy)
    
    variance_explained = pca.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(variance_explained)
    
    print(f"\nVariância explicada (PCA):")
    for i in range(n_componentes_pca):
        print(f"  PC{i+1}: {variance_explained[i]:.2f}% (acumulada: {cumulative_variance[i]:.2f}%)")
    
    # Scree plot PCA
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, n_componentes_pca + 1), variance_explained, 
            'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Componente Principal', fontsize=12)
    ax.set_ylabel('Variância Explicada (%)', fontsize=12)
    ax.set_title('Scree Plot - PCA', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('1_pca_scree_plot.png', bbox_inches='tight')
    print("\n✓ Gráfico salvo: 1_pca_scree_plot.png")
    plt.close()
    
    for i in range(n_componentes_pca):
        df_modelo[f'pca_comp_{i+1}'] = X_cluster[:, i]
    
    metodo_reducao = 'PCA'
    df_cluster = df_modelo.copy()

# =============================================================================
# 6. ANÁLISE DE CLUSTER
# =============================================================================

print("\n" + "="*80)
print("[6/8] ANÁLISE DE CLUSTER")
print("="*80)

# Padronizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Determinar número ótimo de clusters (k=2 a k=10)
print("\nDeterminando número ótimo de clusters...")

k_max = min(10, len(df_cluster) // 10)  # Máximo: N/10
k_range = range(2, k_max + 1)

inertias = []
silhouette_scores = []
davies_bouldin_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))

# Visualizar métricas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de Clusters (k)', fontsize=12)
axes[0].set_ylabel('Inércia (WCSS)', fontsize=12)
axes[0].set_title('Método do Cotovelo', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Número de Clusters (k)', fontsize=12)
axes[1].set_ylabel('Coeficiente de Silhueta', fontsize=12)
axes[1].set_title('Análise de Silhueta', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

axes[2].plot(k_range, davies_bouldin_scores, 'go-', linewidth=2, markersize=8)
axes[2].set_xlabel('Número de Clusters (k)', fontsize=12)
axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12)
axes[2].set_title('Davies-Bouldin (menor = melhor)', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('2_selecao_clusters.png', bbox_inches='tight')
print("✓ Gráfico salvo: 2_selecao_clusters.png")
plt.close()

# Selecionar k ótimo (maior silhueta)
k_otimo = list(k_range)[np.argmax(silhouette_scores)]
silhueta_otima = max(silhouette_scores)
db_otimo = davies_bouldin_scores[np.argmax(silhouette_scores)]

print(f"\nNúmero ótimo de clusters: k = {k_otimo}")
print(f"Coeficiente de Silhueta: {silhueta_otima:.3f}")
print(f"Davies-Bouldin Index: {db_otimo:.3f}")

if silhueta_otima < 0.25:
    print("⚠️ Silhueta muito baixa: estrutura fraca")
elif silhueta_otima < 0.5:
    print("✓ Silhueta razoável: estrutura moderada")
else:
    print("✓ Silhueta boa: estrutura bem definida")

# Clustering final
kmeans_final = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans_final.fit_predict(X_scaled)

# Estatísticas por cluster
print("\n" + "-"*80)
print("CARACTERÍSTICAS DOS CLUSTERS")
print("-"*80)

# Tamanho dos clusters
tamanhos = df_cluster['cluster'].value_counts().sort_index()
print("\nTamanho dos clusters:")
for cluster, n in tamanhos.items():
    print(f"  Cluster {cluster}: {n} acidentes ({n/len(df_cluster)*100:.1f}%)")

# Estatísticas numéricas
cluster_stats_num = df_cluster.groupby('cluster').agg({
    'total_vitimas': ['count', 'mean', 'std'],
    'n_mortos': ['sum', 'mean'],
    'n_feridos_graves': ['sum', 'mean'],
    'n_feridos_leves': ['sum', 'mean'],
    'indice_gravidade': ['mean', 'std']
}).round(2)

print("\nEstatísticas numéricas por cluster:")
print(cluster_stats_num)

# Perfil categórico
print("\n" + "-"*80)
print("PERFIL CATEGÓRICO DOS CLUSTERS")
print("-"*80)

for var in variaveis_categoricas:
    print(f"\n{var}:")
    perfil = pd.crosstab(df_cluster['cluster'], df_cluster[var], 
                         normalize='index') * 100
    print(perfil.round(1))

# Visualizar clusters
if X_cluster.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for cluster in range(k_otimo):
        mask = df_cluster['cluster'] == cluster
        ax.scatter(X_cluster[mask, 0], X_cluster[mask, 1],
                  label=f'Cluster {cluster} (n={mask.sum()})',
                  s=80, alpha=0.6, edgecolors='black', linewidths=1)
    
    ax.set_xlabel(f'{metodo_reducao} - Dimensão 1', fontsize=12)
    ax.set_ylabel(f'{metodo_reducao} - Dimensão 2', fontsize=12)
    ax.set_title(f'Clusters (k={k_otimo})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('3_clusters_visualizacao.png', bbox_inches='tight')
    print("\n✓ Gráfico salvo: 3_clusters_visualizacao.png")
    plt.close()

# =============================================================================
# 7. TESTES ESTATÍSTICOS (ANOVA/KRUSKAL-WALLIS)
# =============================================================================

print("\n" + "="*80)
print("[7/8] TESTES ESTATÍSTICOS")
print("="*80)

# Verificar tamanho dos clusters
n_por_cluster = df_cluster['cluster'].value_counts().sort_index()
n_min = n_por_cluster.min()

print(f"\nTamanho mínimo de cluster: {n_min}")
if n_min < 10:
    print("⚠️ Alguns clusters têm poucos casos. Resultados devem ser interpretados com cautela.")

# Variáveis dependentes
variaveis_dep = {
    'total_vitimas': 'Total de Vítimas',
    'n_mortos': 'Número de Mortos',
    'n_feridos_graves': 'Feridos Graves',
    'n_feridos_leves': 'Feridos Leves',
    'indice_gravidade': 'Índice de Gravidade'
}

df_teste = df_cluster[['cluster'] + list(variaveis_dep.keys())].copy()

# Executar testes
print("\n" + "-"*80)
print("TESTES DE HIPÓTESE")
print("-"*80)

resultados = []

for var, nome in variaveis_dep.items():
    # Preparar grupos
    grupos = [df_teste[df_teste['cluster'] == c][var].dropna().values
              for c in sorted(df_teste['cluster'].unique())]
    
    # Teste de normalidade (Shapiro-Wilk) para cada grupo
    normalidade = []
    for i, grupo in enumerate(grupos):
        if len(grupo) >= 3:
            _, p_shapiro = shapiro(grupo)
            normalidade.append(p_shapiro > 0.05)
        else:
            normalidade.append(False)
    
    todos_normais = all(normalidade)
    
    # Escolher teste apropriado
    if todos_normais and n_min >= 10:
        # ANOVA paramétrica
        f_stat, p_value = f_oneway(*grupos)
        teste = 'ANOVA'
    else:
        # Kruskal-Wallis não-paramétrico
        h_stat, p_value = kruskal(*grupos)
        f_stat = h_stat
        teste = 'Kruskal-Wallis'
    
    # Significância
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
        'Teste': teste,
        'Estatística': f"{f_stat:.3f}",
        'p-valor': f"{p_value:.4f}",
        'Sig.': sig
    })

df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False))
print("\nLegenda: *** p<0.001, ** p<0.01, * p<0.05, ns não significativo")

# Correção de Bonferroni
p_values = [float(r['p-valor']) for r in resultados]
reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

print("\n" + "-"*80)
print("CORREÇÃO DE BONFERRONI")
print("-"*80)
print(f"Nível α original: 0.05")
print(f"Nível α ajustado: {0.05/len(p_values):.4f}")

bonferroni_results = []
for i, (var_nome, p_orig, p_corr, rej) in enumerate(zip(
    [r['Variável'] for r in resultados], 
    p_values, 
    p_corrected, 
    reject
)):
    bonferroni_results.append({
        'Variável': var_nome,
        'p-valor': f"{p_orig:.4f}",
        'p-ajustado': f"{p_corr:.4f}",
        'Significativo': 'Sim' if rej else 'Não'
    })

df_bonferroni = pd.DataFrame(bonferroni_results)
print("\n" + df_bonferroni.to_string(index=False))

# Testes post-hoc para variáveis significativas
print("\n" + "-"*80)
print("TESTES POST-HOC (Tukey HSD)")
print("-"*80)

variaveis_sig = [var for var, resultado in zip(variaveis_dep.keys(), resultados)
                 if resultado['Sig.'] != 'ns']

if len(variaveis_sig) > 0:
    for var in variaveis_sig:
        print(f"\n{variaveis_dep[var]}:")
        try:
            tukey = pairwise_tukeyhsd(endog=df_teste[var], 
                                       groups=df_teste['cluster'], 
                                       alpha=0.05)
            print(tukey)
        except Exception as e:
            print(f"  Não foi possível calcular: {e}")
else:
    print("\nNenhuma variável apresentou diferença significativa entre clusters.")

# Visualizar distribuições
n_vars = len(variaveis_dep)
n_cols = 3
n_rows = (n_vars + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
axes = axes.ravel() if n_vars > 1 else [axes]

for idx, (var, nome) in enumerate(variaveis_dep.items()):
    df_teste.boxplot(column=var, by='cluster', ax=axes[idx])
    axes[idx].set_title(nome, fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Cluster', fontsize=11)
    axes[idx].set_ylabel(nome, fontsize=11)
    axes[idx].get_figure().suptitle('')
    
    # Adicionar resultado do teste
    resultado = resultados[idx]
    axes[idx].text(0.5, 0.98, 
                   f"{resultado['Teste']}: {resultado['Estatística']}, p={resultado['p-valor']} {resultado['Sig.']}",
                   transform=axes[idx].transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)

# Ocultar subplots vazios
for idx in range(n_vars, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Distribuição das Variáveis por Cluster',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('4_testes_estatisticos.png', bbox_inches='tight')
print("\n✓ Gráfico salvo: 4_testes_estatisticos.png")
plt.close()

# =============================================================================
# 8. EXPORTAR RESULTADOS E RELATÓRIO
# =============================================================================

print("\n" + "="*80)
print("[8/8] EXPORTANDO RESULTADOS")
print("="*80)

# Exportar datasets
df_analise.to_csv('1_acidentes_agregados.csv', index=False)
print("✓ 1_acidentes_agregados.csv")

df_cluster.to_csv('2_acidentes_com_clusters.csv', index=False)
print("✓ 2_acidentes_com_clusters.csv")

cluster_stats_num.to_csv('3_estatisticas_clusters.csv')
print("✓ 3_estatisticas_clusters.csv")

df_resultados.to_csv('4_resultados_testes.csv', index=False)
print("✓ 4_resultados_testes.csv")

df_bonferroni.to_csv('5_bonferroni_correcao.csv', index=False)
print("✓ 5_bonferroni_correcao.csv")

# Perfil categórico
perfis_consolidados = []
for var in variaveis_categoricas:
    perfil = pd.crosstab(df_cluster['cluster'], df_cluster[var], 
                         normalize='index') * 100
    perfil['variavel'] = var
    perfil = perfil.reset_index()
    perfis_consolidados.append(perfil)

pd.concat(perfis_consolidados).to_csv('6_perfil_categorico_clusters.csv', index=False)
print("✓ 6_perfil_categorico_clusters.csv")

# Relatório consolidado
with open('RELATORIO_FINAL.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RELATÓRIO DE ANÁLISE MULTIVARIADA\n")
    f.write("Acidentes de Infraestrutura - PRF 2024 - RIDE-DF\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. RESUMO DOS DADOS\n")
    f.write("-"*80 + "\n")
    f.write(f"Total de vítimas no banco: {len(df):,}\n")
    f.write(f"Total de acidentes únicos: {df['id'].nunique():,}\n")
    f.write(f"Acidentes de infraestrutura: {len(acidentes_infra):,} ({len(acidentes_infra)/df['id'].nunique()*100:.1f}%)\n")
    f.write(f"Vítimas de acidentes de infraestrutura: {len(df_infra):,} ({len(df_infra)/len(df)*100:.1f}%)\n")
    f.write(f"Média de vítimas/acidente (geral): {len(df)/df['id'].nunique():.2f}\n")
    f.write(f"Média de vítimas/acidente (infraestrutura): {df_analise['total_vitimas'].mean():.2f}\n")
    f.write(f"Acidentes fatais: {(df_analise['n_mortos'] > 0).sum()} ({(df_analise['n_mortos'] > 0).mean()*100:.1f}%)\n\n")
    
    f.write("2. ANÁLISE DE REDUÇÃO DIMENSIONAL\n")
    f.write("-"*80 + "\n")
    f.write(f"Método utilizado: {metodo_reducao}\n")
    f.write(f"Variância explicada (primeiras 3 dimensões): {cumulative_variance[2]:.1f}%\n\n")
    
    f.write("3. ANÁLISE DE CLUSTER\n")
    f.write("-"*80 + "\n")
    f.write(f"Número de clusters: {k_otimo}\n")
    f.write(f"Coeficiente de Silhueta: {silhueta_otima:.3f}\n")
    f.write(f"Davies-Bouldin Index: {db_otimo:.3f}\n\n")
    
    f.write("Distribuição dos clusters:\n")
    for cluster, n in tamanhos.items():
        f.write(f"  Cluster {cluster}: {n} acidentes ({n/len(df_cluster)*100:.1f}%)\n")
    f.write("\n")
    
    f.write("4. TESTES ESTATÍSTICOS\n")
    f.write("-"*80 + "\n")
    f.write(df_resultados.to_string(index=False))
    f.write("\n\n")
    
    f.write("5. CORREÇÃO DE BONFERRONI\n")
    f.write("-"*80 + "\n")
    f.write(f"Nível α ajustado: {0.05/len(p_values):.4f}\n\n")
    f.write(df_bonferroni.to_string(index=False))
    f.write("\n\n")
    
    f.write("6. CONCLUSÕES\n")
    f.write("-"*80 + "\n")
    f.write("- Acidentes de infraestrutura representam 10.9% dos acidentes mas 21.9% das vítimas\n")
    f.write("- Acidentes de infraestrutura são mais graves que a média (17.3 vs 8.7 vítimas/acidente)\n")
    f.write(f"- Identificados {k_otimo} padrões distintos de acidentes\n")
    
    n_sig = sum([1 for r in resultados if r['Sig.'] != 'ns'])
    if n_sig > 0:
        f.write(f"- {n_sig} variável(is) apresentou(aram) diferença significativa entre clusters\n")
    else:
        f.write("- Nenhuma variável apresentou diferença estatisticamente significativa entre clusters\n")
        f.write("  (limitação do tamanho amostral e alta variabilidade intra-cluster)\n")
    
    f.write("\n7. LIMITAÇÕES\n")
    f.write("-"*80 + "\n")
    f.write(f"- Tamanho amostral: N={len(df_analise)} acidentes\n")
    f.write(f"- Menor cluster: {n_min} acidentes\n")
    f.write("- Alta variabilidade intra-cluster\n")
    f.write("- Análise restrita à RIDE-DF e ano 2024\n")

print("✓ RELATORIO_FINAL.txt")

print("\n" + "="*80)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("="*80)
print("\nArquivos gerados:")
print("  Gráficos (4):")
print(f"    - 1_{metodo_reducao.lower()}_scree_plot.png")
print("    - 2_selecao_clusters.png")
print("    - 3_clusters_visualizacao.png")
print("    - 4_testes_estatisticos.png")
print("  Tabelas (6):")
print("    - 1_acidentes_agregados.csv")
print("    - 2_acidentes_com_clusters.csv")
print("    - 3_estatisticas_clusters.csv")
print("    - 4_resultados_testes.csv")
print("    - 5_bonferroni_correcao.csv")
print("    - 6_perfil_categorico_clusters.csv")
print("  Relatório:")
print("    - RELATORIO_FINAL.txt")

print("\n" + "="*80)
print("MELHORIAS IMPLEMENTADAS:")
print("="*80)
print("✓ Análise no nível de ACIDENTE (N=236) ao invés de segmento (N=22)")
print("✓ Remoção de variáveis sem variabilidade (pista_tipo)")
print("✓ Escolha automática entre ANOVA e Kruskal-Wallis")
print("✓ Testes post-hoc (Tukey HSD) para variáveis significativas")
print("✓ Correção de Bonferroni para comparações múltiplas")
print("✓ Davies-Bouldin Index adicional para validação de clusters")
print("✓ Relatório consolidado com todas as métricas")
print("✓ Tratamento robusto de erros e fallback (PCA se MCA falhar)")
