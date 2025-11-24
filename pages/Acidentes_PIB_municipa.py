# importações e configuração do path

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
import unicodedata

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
                    SELECT ano_pib, codigo_municipio_dv, vl_pib_per_capta
    FROM public.pib_municipios
    WHERE ano_pib = (SELECT MAX(ano_pib) FROM public.pib_municipios)
""")

df_pib.rename(columns={'codigo_municipio_dv': 'codigo_ibge_municipio'}, inplace=True)
df_pib['codigo_ibge_municipio'] = df_pib['codigo_ibge_municipio'].astype(str)
df_pib['vl_pib_per_capta'] = pd.to_numeric(df_pib['vl_pib_per_capta'], errors='coerce')

df_municipio_base = fetch_data("""
    SELECT 
        codigo_ibge_municipio_dv, nome_municipio, cd_uf
    FROM public.municipios
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




# =========================  Dashboard  =========================
st.header("Análise de Acidentes de Trânsito por Município e PIB")
st.write(f"Acidentes agrupados: {df_acidentes_count.shape[0]} municípios.")

st.subheader("Top 5 Municípios com Maior Taxa de Acidentes")
st.dataframe(df_exibicao.sort_values(by='taxa_acidentes_100k', ascending=False).head(5))

st.subheader("Correlação entre PIB per Capita e Taxa de Acidentes por 100 mil Habitantes")

fig, ax = plt.subplots(figsize=(12, 8))

# Plotar pontos
sns.regplot(
    x='vl_pib_per_capta',
    y='taxa_acidentes_100k',
    ax=ax,
    data=df_analise,
    scatter_kws={'s': 100, 'alpha': 0.7, 'color': 'steelblue'}, # Estilo dos pontos
    line_kws={'color': 'red', 'linestyle': '--'} # Linha de tendência
)

# Adicionar Rótulos (Nomes dos Municípios) aos pontos
# Isso ajuda a identificar Brasília e outros destaques
for i in range(df_analise.shape[0]):
    plt.text(
        df_analise.vl_pib_per_capta.iloc[i] + 500, # Ajuste leve na posição X
        df_analise.taxa_acidentes_100k.iloc[i],
        df_analise.chave_municipio.iloc[i],
        fontsize=9,
        alpha=0.7
    )

plt.title(f'Relação: Riqueza (PIB per capita) vs Segurança no Trânsito\nCorrelação: {corr_coef:.2f}', fontsize=14)
plt.xlabel('PIB per Capita (R$)', fontsize=12)
plt.ylabel('Acidentes por 100k habitantes', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
st.pyplot(fig)


st.write(f"Coeficiente de Correlação (Pearson): {corr_coef:.4f}.\t P-valor: {p_value:.4f}.\t Interpretação: Correlação {interpretacao} {tipo}")
