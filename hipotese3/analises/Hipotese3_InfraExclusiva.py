# =============================================================================
# MCA E CLUSTER USANDO CAUSAS DE INFRAESTRUTURA COMO VARIÁVEIS DIRETAS

""" Alt+Z para quebrar linhas automaticamente no VSCode

A análise multivariada baseada exclusivamente nas causas de infraestrutura revelou que acidentes agrupados por perfis semelhantes dessas condições formam clusters estatisticamente bem definidos (silhueta 0.92), com perfis de gravidade distintos. Isso demonstra que não só as condições isoladas, mas, especialmente, combinações de problemas de infraestrutura (exemplo: falta de acostamento + sinalização mal posicionada + pista escorregadia) estão associadas a acidentes de alta severidade. Esses agrupamentos são fundamentais para priorizar intervenções e políticas públicas dirigidas a perfis de risco real em vez de apenas condições isoladas.
"""
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import prince
import os

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# =============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =============================================================================

df = pd.read_csv("/home/jobsr/Documents/MultivariadaII/analises/Hipotese3_Resultados/1_acidentes_agregados.csv")
# OBS: Tem que ter rodado o script Hipotese3.Infra.py antes!

causas_infraestrutura = [
    'Iluminação deficiente', 'Demais falhas na via', 'Pista esburacada', 'Falta de acostamento',
    'Acesso irregular', 'Acumulo de água sobre o pavimento', 'Acumulo de areia ou detritos sobre o pavimento',
    'Acumulo de óleo sobre o pavimento', 'Falta de elemento de contenção que evite a saída do leito carroçável',
    'Restrição de visibilidade em curvas verticais', 'Restrição de visibilidade em curvas horizontais',
    'Sistema de drenagem ineficiente', 'Declive acentuado', 'Curva acentuada', 'Sinalização mal posicionada',
    'Desvio temporário', 'Ausência de sinalização', 'Afundamento ou ondulação no pavimento', 'Acostamento em desnível',
    'Deficiência do Sistema de Iluminação/Sinalização', 'Pista Escorregadia'
]

# Gera as dummies caso ainda não estejam no dataset
for causa in causas_infraestrutura:
    var = f"causa_{causa.lower().replace(' ', '_').replace('/', '_').replace(';','').replace('-','_')}"
    if var not in df.columns:
        df[var] = (df['causa_acidente'] == causa).astype(int)

causa_vars = [f"causa_{c.lower().replace(' ', '_').replace('/', '_').replace(';','').replace('-','_')}" for c in causas_infraestrutura]

# Remover causas sem variabilidade
valid_vars = [col for col in causa_vars if df[col].sum() > 0]
print(f"Variáveis de infraestrutura com presença: {valid_vars}")

df_model = df[valid_vars].copy()
print(f"Shape para MCA: {df_model.shape}")

# =============================================================================
# 2. MCA SOBRE AS CAUSAS DE INFRAESTRUTURA
# =============================================================================

mca = prince.MCA(n_components=min(10, len(valid_vars)), n_iter=5, random_state=42)
mca = mca.fit(df_model)
mca_coords = mca.transform(df_model)

n_componentes = min(3, len(valid_vars))
if hasattr(mca, "explained_inertia_"):
    print("Variância explicada (cada dim):", mca.explained_inertia_[:n_componentes])
    print("Variância acumulada (3 dims):", np.sum(mca.explained_inertia_[:n_componentes]))
else:
    variancia_expl = mca.eigenvalues_ / mca.eigenvalues_.sum()
    print("Variância explicada (cada dim):", variancia_expl[:n_componentes])
    print("Variância acumulada (3 dims):", np.sum(variancia_expl[:n_componentes]))

# Scree plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.arange(1, len(variancia_expl) + 1), 100 * variancia_expl, 
        'bo-', linewidth=2, markersize=7)
ax.set_title('Scree Plot - MCA Infraestrutura')
ax.set_xlabel('Dimensão')
ax.set_ylabel('Variância Explicada (%)')
plt.tight_layout()
plt.savefig("mca_infraestrutra_scree_plot.png")
plt.close()

for i in range(n_componentes):
    df[f'mca_infra_{i+1}'] = mca_coords.iloc[:, i].values

# =============================================================================
# 3. CLUSTERIZAÇÃO USANDO MCA (INFRAESTRUTURA)
# =============================================================================

X_cluster = mca_coords.iloc[:, :n_componentes].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Encontrar k ótimo (2 a 6 é bom para N pequeno)
k_opts = range(2, min(7, int(len(df)/10) + 1))
silh = []
dbidx = []

for k in k_opts:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silh.append(silhouette_score(X_scaled, labels))
    dbidx.append(davies_bouldin_score(X_scaled, labels))

k_otimo = k_opts[np.argmax(silh)]
print(f"k ótimo: {k_otimo} (Silhueta={max(silh):.2f})")

# Gráfico de seleção de clusters
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(k_opts, silh, 'ro-')
axes[0].set_title('Silhueta (quanto maior, melhor)')
axes[0].set_xlabel('Clusters (k)')
axes[0].set_ylabel('Coef. de Silhueta')
axes[1].plot(k_opts, dbidx, 'bo-')
axes[1].set_title('Davies-Bouldin (quanto menor, melhor)')
axes[1].set_xlabel('Clusters')
axes[1].set_ylabel('DB Index')
plt.tight_layout()
plt.savefig("infra_selecao_clusters.png")
plt.close()

# Clustering final
kmeans_final = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
df['infra_cluster'] = kmeans_final.fit_predict(X_scaled)

# MCA Mapping
if n_componentes >= 2:
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x=mca_coords.iloc[:,0], y=mca_coords.iloc[:,1], 
                    hue=df['infra_cluster'], palette='tab10', s=70)
    ax.set_xlabel('MCA Infraestrutura - Dim 1')
    ax.set_ylabel('MCA Infraestrutura - Dim 2')
    ax.set_title(f'Clusters por Perfil de Infraestrutura (k={k_otimo})')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("infra_clusters_mca_mapping.png")
    plt.close()

# =============================================================================
# 4. RESUMO E EXPORTAÇÃO
# =============================================================================

# Cluster profile table
perfil_causas = df.groupby('infra_cluster')[valid_vars].mean().round(2)
perfil_causas.to_csv("perfil_infra_clustes.csv")
print("✓ perfil_infra_clustes.csv")

perfil_stats = df.groupby('infra_cluster').agg({
    'indice_gravidade': ['mean', 'median', 'std', 'count'],
    'n_mortos': ['mean', 'median', 'std'],
    'n_feridos_graves': ['mean', 'median', 'std'],
})
perfil_stats.to_csv("gravidade_infra_clusters.csv")
print("✓ gravidade_infra_clusters.csv")

# Visual: boxplot gravidade por cluster
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(data=df, x='infra_cluster', y='indice_gravidade', palette='tab10')
ax.set_title('Distribuição da Gravidade por Cluster de Infraestrutura')
ax.set_xlabel('Infraestrutura - Cluster')
ax.set_ylabel('Índice de Gravidade')
plt.tight_layout()
plt.savefig("infra_cluster_boxplot_gravidade.png")
plt.close()

print("✓ infra_cluster_boxplot_gravidade.png")
print("Análise MCA + clusters SÓ com variáveis de condição de infraestrutura concluída!")
