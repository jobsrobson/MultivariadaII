# =============================================================================
# ANÁLISE DE GRAVIDADE POR TIPO DE INFRAESTRUTURA (acidentes PRF)
# Cada linha = acidente agregado (como no df_analise do pipeline anterior)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 300

# =============================================================================
# 1. CARREGAMENTO DOS DADOS
# =============================================================================

# Altere para o caminho do seu arquivo já agregado por acidente (uma linha = um acidente)
df = pd.read_csv("/home/jobsr/Documents/MultivariadaII/analises/Hipotese3_Resultados/1_acidentes_agregados.csv")

# =============================================================================
# 2. LISTA DE CAUSAS DE INFRAESTRUTURA
# =============================================================================

causas_infraestrutura = [
    'Iluminação deficiente', 'Demais falhas na via', 'Pista esburacada', 'Falta de acostamento',
    'Acesso irregular', 'Acumulo de água sobre o pavimento', 'Acumulo de areia ou detritos sobre o pavimento',
    'Acumulo de óleo sobre o pavimento', 'Falta de elemento de contenção que evite a saída do leito carroçável',
    'Restrição de visibilidade em curvas verticais', 'Restrição de visibilidade em curvas horizontais',
    'Sistema de drenagem ineficiente', 'Declive acentuado', 'Curva acentuada', 'Sinalização mal posicionada',
    'Desvio temporário', 'Ausência de sinalização', 'Afundamento ou ondulação no pavimento', 'Acostamento em desnível',
    'Deficiência do Sistema de Iluminação/Sinalização', 'Pista Escorregadia'
]

# =============================================================================
# 3. CRIAR VARIÁVEIS INDIVIDUAIS DE CAUSA
# =============================================================================

for causa in causas_infraestrutura:
    var = f"causa_{causa.lower().replace(' ', '_').replace('/', '_').replace(';','').replace('-','_')}"
    df[var] = (df['causa_acidente'] == causa).astype(int)

# Lista dos nomes das variáveis criadas
causa_vars = [f"causa_{c.lower().replace(' ', '_').replace('/', '_').replace(';','').replace('-','_')}" for c in causas_infraestrutura]

# =============================================================================
# 4. ESTATÍSTICAS E RANKING
# =============================================================================

resultados = []
for causa, var in zip(causas_infraestrutura, causa_vars):
    grupo_1 = df[df[var] == 1]['indice_gravidade']
    grupo_0 = df[df[var] == 0]['indice_gravidade']
    n1 = len(grupo_1)
    n0 = len(grupo_0)
    # Só considerar se houver amostra minimamente robusta (≥4)
    if n1 >= 4:
        media_1 = grupo_1.mean()
        std_1 = grupo_1.std()
        try:
            stat, p = mannwhitneyu(grupo_1, grupo_0, alternative='two-sided')
        except ValueError:
            p = np.nan
        resultados.append({
            "Causa": causa,
            "N_acidentes": n1,
            "Gravidade_média": media_1,
            "Desvio_Padrão": std_1,
            "p_MannWhitney": p
        })

# Tabela de resumo ordenada por gravidade média
df_result = pd.DataFrame(resultados).sort_values("Gravidade_média", ascending=False)

df_result.to_csv("gravidade_por_causa.csv", index=False)
print("✓ gravidade_por_causa.csv")

# =============================================================================
# 5. GRÁFICOS - BOXPLOTS POR CAUSA
# =============================================================================

os.makedirs("causas_graficos", exist_ok=True)
for causa, var in zip(causas_infraestrutura, causa_vars):
    grupo_1 = df[df[var] == 1]['indice_gravidade']
    if len(grupo_1) >= 4:
        # Evitar gráficos com casos únicos (sem sentido estatístico)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(data=df, x=df[var], y='indice_gravidade', ax=ax)
        ax.set_xticklabels(['Não', 'Sim'])
        ax.set_xlabel(f"Acidentes com {causa.lower()}")
        ax.set_ylabel('Índice de Gravidade')
        ax.set_title(f"Gravidade por Presença de '{causa}'")
        plt.tight_layout()
        plt.savefig(f"causas_graficos/boxplot_{var}.png")
        plt.close()

print("✓ Gráficos individuais salvos na pasta causas_graficos/")

# =============================================================================
# 6. GRÁFICO DE RANKING (TOP 8 MAIS GRAVES)
# =============================================================================

top = df_result.sort_values("Gravidade_média", ascending=False).head(8)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=top, x='Gravidade_média', y='Causa', palette='Reds_r', orient='h', ax=ax)
ax.set_xlabel('Índice de Gravidade Médio')
ax.set_ylabel('Causa de Infraestrutura')
ax.set_title('Top 8 Condições de Infraestrutura por Gravidade Média')
for i, v in enumerate(top['Gravidade_média']):
    ax.text(v + 0.3, i, f"{v:.1f}", va='center', fontsize=9)
plt.tight_layout()
plt.savefig("ranking_top8_gravidade_causa.png")
plt.close()
print("✓ ranking_top8_gravidade_causa.png")

# =============================================================================
# 7. TABELA COMPACTA PARA RELATÓRIO
# =============================================================================

# Selecionar colunas importantes e salvar
df_result[['Causa', 'N_acidentes', 'Gravidade_média', 'p_MannWhitney']].to_csv("gravidade_causa_compacta.csv", index=False)
print("✓ gravidade_causa_compacta.csv")

print("Análise concluída. Veja: gravidade_por_causa.csv, ranking_top8_gravidade_causa.png, causas_graficos/")
