import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.multivariate.manova import MANOVA

from app import fetch_data

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Frotas Mais Potentes Geram Mais Acidentes",
    page_icon="1️⃣",
    layout="wide"
)

# --- 2. CARREGAMENTO DOS DADOS ---

dtypes_otimizados = {
    'data_inversa':           'object',
    'dia_semana':             'category',
    'horario':                'object',
    'uf':                     'category',
    'br':                     'Int64',
    'municipio':              'object',
    'ano_fabricacao_veiculo': 'Int64',
    'tipo_envolvido':         'category',
    'estado_fisico':          'category',
    'idade':                  'Int64',
    'sexo':                   'category',
    'ilesos':                 'Int64',
    'feridos_leves':          'Int64',
    'feridos_graves':         'Int64',
    'mortos':                 'Int64'
}

ac = fetch_data("SELECT * FROM public.acidente_transito")

print(ac['municipio'].unique())
ac['km'] = ac['km'].astype(str).str.replace(',', '.')
ac['km'] = pd.to_numeric(ac['km'], errors='coerce')

for col, dtype in dtypes_otimizados.items():
    ac[col] = ac[col].astype(dtype)

ac['data_inversa'] = pd.to_datetime(ac['data_inversa'], errors='coerce')

bins = [-float('inf'), 1989, 1999, 2009, 2019, float('inf')]
labels = ['Antigos (<1990)', 'Anos 90', 'Anos 2000', 'Anos 2010', 'Modernos (2020+)']
ac['categoria_ano'] = pd.cut(ac['ano_fabricacao_veiculo'], bins=bins, labels=labels)
print(ac['categoria_ano'].value_counts().sort_index())

# --- 3. Página Streamlit ---

st.markdown("<h2>Hipótese 1: Frotas Mais Novas Geram Menos Acidentes</h2>", unsafe_allow_html=True)

st.markdown("""
Esta análise tem como objetivo investigar se existe uma relação entre a idade dos veículos 
envolvidos em acidentes e a gravidade desses eventos. A pergunta central que buscamos responder é: 
**veículos mais novos estão associados a menos mortes e feridos em acidentes de trânsito?**
""")

st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# SEÇÃO 1: VISUALIZAÇÃO - Número de Acidentes por Categoria
# =============================================================================
with st.container(border=True):
    st.markdown("### Distribuição dos Acidentes por Categoria de Ano de Fabricação")
    
    st.markdown("""
    Antes de partirmos para os testes estatísticos, é importante visualizar como os acidentes 
    se distribuem entre as diferentes categorias de idade dos veículos. O gráfico abaixo nos 
    dá uma primeira impressão dessa distribuição.
    """)
    
    acidentes_por_categoria = ac['categoria_ano'].value_counts().sort_index().reset_index()
    acidentes_por_categoria.columns = ['Categoria Ano', 'Número de Acidentes']
    
    fig = px.bar(
        acidentes_por_categoria,
        x='Categoria Ano',
        y='Número de Acidentes',
        title='Número de Acidentes por Categoria de Ano de Fabricação',
        labels={'Número de Acidentes': 'Número de Acidentes', 'Categoria Ano': 'Categoria Ano de Fabricação'},
        text='Número de Acidentes'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis=dict(range=[0, acidentes_por_categoria['Número de Acidentes'].max() * 1.1]))
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Resumo dos Dados")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_acidentes = acidentes_por_categoria['Número de Acidentes'].sum()
        st.metric("Total de Registros", f"{total_acidentes:,}")
    with col2:
        categoria_mais_acidentes = acidentes_por_categoria.loc[
            acidentes_por_categoria['Número de Acidentes'].idxmax(), 'Categoria Ano'
        ]
        st.metric("Categoria com Mais Acidentes", categoria_mais_acidentes)
    with col3:
        categoria_menos_acidentes = acidentes_por_categoria.loc[
            acidentes_por_categoria['Número de Acidentes'].idxmin(), 'Categoria Ano'
        ]
        st.metric("Categoria com Menos Acidentes", categoria_menos_acidentes)

# =============================================================================
# SEÇÃO 2: VISUALIZAÇÃO - Severidade por Categoria
# =============================================================================
with st.container(border=True):
    st.markdown("### Severidade dos Acidentes por Categoria de Ano")
    
    st.markdown("""
    Além da quantidade de acidentes, é fundamental analisar a **gravidade** dos mesmos. 
    O gráfico a seguir apresenta o número de pessoas afetadas em cada categoria, 
    divididas por nível de severidade: ilesos, feridos leves, feridos graves e mortos.
    """)
    
    acidentes_severidade = ac.groupby('categoria_ano')[['ilesos', 'feridos_leves', 'feridos_graves', 'mortos']].sum().reset_index()
    acidentes_severidade_melted = acidentes_severidade.melt(
        id_vars='categoria_ano',
        value_vars=['ilesos', 'feridos_leves', 'feridos_graves', 'mortos'],
        var_name='Severidade',
        value_name='Número de Pessoas'
    )
    
    fig2 = px.bar(
        acidentes_severidade_melted,
        x='categoria_ano',
        y='Número de Pessoas',
        color='Severidade',
        title='Pessoas Envolvidas em Acidentes por Categoria de Ano de Fabricação',
        labels={'categoria_ano': 'Categoria Ano de Fabricação', 'Número de Pessoas': 'Número de Pessoas'},
        text='Número de Pessoas'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("""
    Para uma análise mais detalhada, apresentamos abaixo os valores absolutos e as proporções 
    de cada tipo de severidade por categoria de veículo.
    """)
    
    st.markdown("#### Valores Absolutos por Categoria")
    st.dataframe(
        acidentes_severidade.set_index('categoria_ano').style.format("{:,.0f}"),
        use_container_width=True
    )
    
    st.markdown("#### Proporção de Severidade por Categoria")
    acidentes_severidade_pct = acidentes_severidade.copy()
    cols_severidade = ['ilesos', 'feridos_leves', 'feridos_graves', 'mortos']
    acidentes_severidade_pct[cols_severidade] = acidentes_severidade_pct[cols_severidade].div(
        acidentes_severidade_pct[cols_severidade].sum(axis=1), axis=0
    ) * 100
    st.dataframe(
        acidentes_severidade_pct.set_index('categoria_ano').style.format("{:.2f}%"),
        use_container_width=True
    )

# =============================================================================
# SEÇÃO 3: MANOVA - Análise Multivariada
# =============================================================================
with st.container(border=True):
    st.markdown("### Análise Multivariada de Variância (MANOVA)")
    
    st.markdown("""   
    A MANOVA (Multivariate Analysis of Variance) nos permite analisar todas essas variáveis 
    simultaneamente, levando em conta a correlação entre elas. Isso é importante porque 
    um acidente grave pode resultar em combinações diferentes de feridos e mortos, 
    e queremos capturar esse padrão conjunto.
    
    **Hipóteses do teste:**
    - **Hipótese Nula (H₀):** Os vetores de médias são iguais em todas as categorias
    - **Hipótese Alternativa (H₁):** Pelo menos uma categoria apresenta vetor de médias diferente
    """)
    
    cols_manova = ['feridos_leves', 'feridos_graves', 'mortos', 'categoria_ano']
    ac_clean = ac[cols_manova].dropna()
    ac_clean = ac_clean[ac_clean['categoria_ano'].notna()]
    ac_clean['categoria_ano'] = ac_clean['categoria_ano'].cat.remove_unused_categories()
    
    if ac_clean['categoria_ano'].nunique() >= 2 and len(ac_clean) > 0:
        manova = MANOVA.from_formula(
            'feridos_leves + feridos_graves + mortos ~ C(categoria_ano)', 
            data=ac_clean
        )
        results_manova = manova.mv_test()
        
        st.markdown("#### Resultados dos Testes Multivariados")
        
        st.markdown("""
        A MANOVA produz várias estatísticas de teste. O Lambda de Wilks é a mais comumente 
        utilizada e pode ser interpretada da seguinte forma: valores mais próximos de zero 
        indicam maior diferença entre os grupos.
        """)
        
        manova_summary = []
        for effect_name, effect_data in results_manova.results.items():
            stat_df = effect_data['stat']
            for test_name in stat_df.index:
                row = stat_df.loc[test_name]
                manova_summary.append({
                    'Efeito': effect_name,
                    'Teste': test_name,
                    'Valor': row['Value'],
                    'Num DF': row['Num DF'],
                    'Den DF': row['Den DF'],
                    'F Value': row['F Value'],
                    'Pr > F': row['Pr > F']
                })
        
        manova_df = pd.DataFrame(manova_summary)
        st.dataframe(manova_df.style.format({
            'Valor': '{:.4f}',
            'Num DF': '{:.0f}',
            'Den DF': '{:.2f}',
            'F Value': '{:.4f}',
            'Pr > F': '{:.6f}'
        }), use_container_width=True)
        
        wilks_row = manova_df[manova_df['Teste'] == "Wilks' lambda"]
        wilks_p = wilks_row['Pr > F'].values[0]
        wilks_value = wilks_row['Valor'].values[0]
        wilks_f = wilks_row['F Value'].values[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lambda de Wilks", f"{wilks_value:.4f}")
        with col2:
            st.metric("Estatística F", f"{wilks_f:.4f}")
        with col3:
            st.metric("Valor-p", f"{wilks_p:.6f}")
        
        if wilks_p < 0.05:
            st.success(f"""
            **Resultado: Estatisticamente Significativo**
            
            O teste Lambda de Wilks apresentou valor-p de {wilks_p:.6f}, inferior ao nível 
            de significância de 0,05. Portanto, rejeitamos a hipótese nula e concluímos que 
            existe diferença multivariada significativa entre as categorias de ano de fabricação, 
            quando consideramos conjuntamente feridos leves, feridos graves e mortos.
            
            Agora precisamos identificar quais variáveis específicas contribuem para essa diferença.
            """)
            
            st.markdown("---")
            st.markdown("### Análises de Seguimento: ANOVAs Univariadas")
            
            st.markdown("""
            Como a MANOVA foi significativa, realizamos agora ANOVAs univariadas para cada 
            variável dependente separadamente. Isso nos permite identificar quais das três 
            variáveis (feridos leves, feridos graves, mortos) apresentam diferenças significativas 
            entre as categorias.
            
            **Correção de Bonferroni:** Como estamos realizando múltiplos testes (3 ANOVAs), 
            precisamos ajustar nosso nível de significância para controlar a taxa de erro 
            tipo I. A correção de Bonferroni divide o alfa original pelo número de testes.
            """)
            
            dependent_vars = ['feridos_leves', 'feridos_graves', 'mortos']
            num_tests = len(dependent_vars)
            bonferroni_alpha = 0.05 / num_tests
            
            st.info(f"Nível de significância ajustado (Bonferroni): {bonferroni_alpha:.4f} (0,05 dividido por {num_tests} testes)")
            
            for var in dependent_vars:
                var_nome = var.replace('_', ' ').title()
                
                with st.expander(f"Análise para: {var_nome}", expanded=True):
                    model_uni = ols(f'{var} ~ C(categoria_ano)', data=ac_clean).fit()
                    anova_table_uni = sm.stats.anova_lm(model_uni, typ=2)
                    
                    p_value_uni = anova_table_uni['PR(>F)'][0]
                    f_value_uni = anova_table_uni['F'][0]
                    
                    st.markdown("**Tabela ANOVA**")
                    st.dataframe(anova_table_uni.style.format("{:.4f}"), use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Estatística F", f"{f_value_uni:.4f}")
                    with col2:
                        st.metric("Valor-p", f"{p_value_uni:.6f}")
                    
                    if p_value_uni < bonferroni_alpha:
                        st.success(f"""
                        **Resultado: Significativo após correção de Bonferroni**
                        
                        O valor-p de {p_value_uni:.6f} é menor que o alfa corrigido de {bonferroni_alpha:.4f}. 
                        Portanto, existe diferença significativa em **{var_nome}** entre as categorias 
                        de ano de fabricação.
                        """)
                        
                        st.markdown("**Comparações Pareadas (Tukey HSD)**")
                        
                        tukey_results_uni = pairwise_tukeyhsd(
                            endog=ac_clean[var], 
                            groups=ac_clean['categoria_ano'], 
                            alpha=0.05
                        )
                        tukey_df_uni = pd.DataFrame(
                            data=tukey_results_uni.summary().data[1:], 
                            columns=tukey_results_uni.summary().data[0]
                        )
                        st.dataframe(tukey_df_uni, use_container_width=True)
                        
                        pares_sig_uni = tukey_df_uni[tukey_df_uni['reject'] == True]
                        if len(pares_sig_uni) > 0:
                            st.markdown("**Pares com diferenças significativas:**")
                            for _, row in pares_sig_uni.iterrows():
                                diff = float(row['meandiff'])
                                if diff > 0:
                                    st.write(f"- **{row['group1']}** apresenta média inferior a **{row['group2']}** (diferença: {abs(diff):.4f})")
                                else:
                                    st.write(f"- **{row['group1']}** apresenta média superior a **{row['group2']}** (diferença: {abs(diff):.4f})")
                        
                        st.markdown("**Médias por Categoria**")
                        medias_var = ac_clean.groupby('categoria_ano')[var].agg(['mean', 'std', 'count']).round(4)
                        medias_var.columns = ['Média', 'Desvio Padrão', 'N']
                        st.dataframe(medias_var.style.format({'Média': '{:.4f}', 'Desvio Padrão': '{:.4f}', 'N': '{:,.0f}'}), use_container_width=True)
                        
                    else:
                        st.warning(f"""
                        **Resultado: Não significativo após correção de Bonferroni**
                        
                        O valor-p de {p_value_uni:.6f} é maior ou igual ao alfa corrigido de {bonferroni_alpha:.4f}. 
                        Portanto, não há evidências suficientes para afirmar que existe diferença 
                        em **{var_nome}** entre as categorias quando aplicamos a correção para 
                        múltiplas comparações.
                        """)
                        
                        st.markdown("**Médias por Categoria (para referência)**")
                        medias_var = ac_clean.groupby('categoria_ano')[var].agg(['mean', 'std', 'count']).round(4)
                        medias_var.columns = ['Média', 'Desvio Padrão', 'N']
                        st.dataframe(medias_var.style.format({'Média': '{:.4f}', 'Desvio Padrão': '{:.4f}', 'N': '{:,.0f}'}), use_container_width=True)
            
            st.markdown("---")
            st.markdown("### Visão Geral das Médias por Categoria")
            
            st.markdown("""
            Para facilitar a interpretação dos resultados, apresentamos abaixo um resumo 
            das médias de todas as variáveis dependentes por categoria de ano de fabricação.
            """)
            
            medias_df = ac_clean.groupby('categoria_ano')[dependent_vars].mean().round(4)
            st.dataframe(medias_df.style.format("{:.4f}").background_gradient(cmap='YlOrRd', axis=0), use_container_width=True)
            
            st.markdown("#### Comparação Visual das Médias")
            medias_melted = medias_df.reset_index().melt(
                id_vars='categoria_ano',
                value_vars=dependent_vars,
                var_name='Variável',
                value_name='Média'
            )
            fig_medias = px.bar(
                medias_melted,
                x='categoria_ano',
                y='Média',
                color='Variável',
                barmode='group',
                title='Comparação das Médias de Feridos e Mortos por Categoria de Ano',
                labels={'categoria_ano': 'Categoria de Ano', 'Média': 'Média'}
            )
            st.plotly_chart(fig_medias, use_container_width=True)
            
        else:
            st.warning(f"""
            **Resultado: Não Significativo**
            
            O teste Lambda de Wilks apresentou valor-p de {wilks_p:.6f}, que é maior ou igual 
            ao nível de significância de 0,05. Portanto, não rejeitamos a hipótese nula.
            
            Isso significa que não encontramos evidências estatísticas suficientes para afirmar 
            que existe diferença multivariada significativa entre as categorias de ano de fabricação, 
            quando consideramos conjuntamente feridos leves, feridos graves e mortos.
            
            Como a MANOVA não foi significativa, não é apropriado realizar análises de seguimento 
            (ANOVAs univariadas), pois isso aumentaria o risco de encontrar resultados 
            falso-positivos.
            """)
            
            st.markdown("#### Médias por Categoria (para referência)")
            dependent_vars = ['feridos_leves', 'feridos_graves', 'mortos']
            medias_df = ac_clean.groupby('categoria_ano')[dependent_vars].mean().round(4)
            st.dataframe(medias_df.style.format("{:.4f}"), use_container_width=True)
            
    else:
        st.error("""
        **Dados insuficientes para a análise**
        
        Não foi possível executar a MANOVA porque os dados não atendem aos requisitos mínimos. 
        Verifique se existem pelo menos duas categorias com observações válidas.
        """)

# =============================================================================
# SEÇÃO 4: CONCLUSÃO FINAL
# =============================================================================
with st.container(border=True):
    st.markdown("### Conclusão da Análise")
    
    # Recuperar os resultados para construir a conclusão dinamicamente
    cols_manova = ['feridos_leves', 'feridos_graves', 'mortos', 'categoria_ano']
    ac_conclusao = ac[cols_manova].dropna()
    ac_conclusao = ac_conclusao[ac_conclusao['categoria_ano'].notna()]
    ac_conclusao['categoria_ano'] = ac_conclusao['categoria_ano'].cat.remove_unused_categories()
    
    # Calcular médias para contextualizar
    medias_conclusao = ac_conclusao.groupby('categoria_ano')[['feridos_leves', 'feridos_graves', 'mortos']].mean()
    
    # Identificar categoria com maior e menor média de mortos
    cat_maior_mortos = medias_conclusao['mortos'].idxmax()
    cat_menor_mortos = medias_conclusao['mortos'].idxmin()
    media_maior = medias_conclusao.loc[cat_maior_mortos, 'mortos']
    media_menor = medias_conclusao.loc[cat_menor_mortos, 'mortos']
    
    # Verificar se MANOVA foi significativa (recalcular para ter o valor)
    try:
        manova_conclusao = MANOVA.from_formula(
            'feridos_leves + feridos_graves + mortos ~ C(categoria_ano)', 
            data=ac_conclusao
        )
        results_conclusao = manova_conclusao.mv_test()
        
        manova_summary_conc = []
        for effect_name, effect_data in results_conclusao.results.items():
            stat_df = effect_data['stat']
            for test_name in stat_df.index:
                row = stat_df.loc[test_name]
                manova_summary_conc.append({
                    'Teste': test_name,
                    'Pr > F': row['Pr > F']
                })
        
        manova_df_conc = pd.DataFrame(manova_summary_conc)
        wilks_p_conc = manova_df_conc[manova_df_conc['Teste'] == "Wilks' lambda"]['Pr > F'].values[0]
        manova_significativa = wilks_p_conc < 0.05
    except:
        manova_significativa = False
        wilks_p_conc = None
    
    if manova_significativa:
        st.markdown(f"""
        Após uma análise estatística detalhada, podemos afirmar com confiança que **existe uma 
        relação significativa entre a idade dos veículos e a gravidade dos acidentes de trânsito**.
        
        A análise multivariada (MANOVA) confirmou que as categorias de ano de fabricação dos 
        veículos apresentam diferenças significativas quando consideramos conjuntamente o número 
        de feridos leves, feridos graves e mortos (valor-p = {wilks_p_conc:.6f}).
        
        **Principais descobertas:**
        
        Os dados revelam que veículos da categoria **{cat_maior_mortos}** apresentam a maior 
        média de mortos por acidente ({media_maior:.4f}), enquanto veículos **{cat_menor_mortos}** 
        apresentam a menor média ({media_menor:.4f}). Essa diferença de {((media_maior - media_menor) / media_menor * 100):.1f}% 
        sugere que a idade do veículo pode ser um fator relevante na gravidade dos acidentes.
        
        **Implicações práticas:**
        
        Esses resultados têm implicações importantes para políticas públicas de segurança no trânsito. 
        Programas de renovação de frota, incentivos fiscais para veículos mais novos com melhores 
        sistemas de segurança, e fiscalização mais rigorosa de veículos antigos podem contribuir 
        para a redução de mortes e ferimentos graves em acidentes de trânsito.
        
        **Limitações do estudo:**
        
        É importante ressaltar que esta análise identifica associações, não relações de causa e efeito. 
        Outros fatores como comportamento do motorista, condições da via, manutenção do veículo e 
        uso de equipamentos de segurança também influenciam a gravidade dos acidentes e não foram 
        controlados nesta análise.
        """)
    else:
        st.markdown(f"""
        Após uma análise estatística detalhada, **não encontramos evidências suficientes** para 
        afirmar que existe uma relação significativa entre a idade dos veículos e a gravidade 
        dos acidentes de trânsito, quando consideramos conjuntamente feridos leves, graves e mortos.
        
        **O que os dados mostram:**
        
        Embora existam diferenças numéricas entre as categorias — veículos **{cat_maior_mortos}** 
        apresentam média de {media_maior:.4f} mortos por acidente, enquanto **{cat_menor_mortos}** 
        apresentam {media_menor:.4f} — essas diferenças não são estatisticamente significativas 
        ao nível de 5%.
        
        **O que isso significa:**
        
        A ausência de significância estatística não significa que a idade do veículo seja irrelevante 
        para a segurança no trânsito. Pode indicar que outros fatores têm maior peso na determinação 
        da gravidade dos acidentes, ou que o tamanho do efeito da idade do veículo é pequeno demais 
        para ser detectado com os dados disponíveis.
        
        **Considerações finais:**
        
        Estudos futuros poderiam explorar interações entre a idade do veículo e outras variáveis, 
        como tipo de acidente, condições climáticas ou características da via, para uma compreensão 
        mais completa dos fatores que influenciam a gravidade dos acidentes de trânsito.
        """)
    
    st.markdown("---")
    
    st.markdown("""
    **Metodologia utilizada nesta análise:**
    
    | Etapa | Técnica | Objetivo |
    |-------|---------|----------|
    | 1 | Análise Exploratória | Visualizar a distribuição dos dados |
    | 2 | MANOVA | Testar diferenças multivariadas entre grupos |
    | 3 | ANOVAs com Bonferroni | Identificar variáveis específicas com diferenças |
    | 4 | Teste de Tukey HSD | Comparar pares de categorias |
    """)

