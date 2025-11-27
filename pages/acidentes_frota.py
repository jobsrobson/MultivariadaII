# ---------------------------------------------
# IMPORTAÇÕES
# ---------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from sklearn.cross_decomposition import CCA
import seaborn as sns
import pandas as pd
import numpy as np

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/toledo-cia/Documents/Projetos/MultivariadaII')

from app import fetch_data

# ---------------------------------------------
# CARREGAMENTO DAS TABELAS (com cache)
# ---------------------------------------------

ac = fetch_data("SELECT * FROM public.acidentes_municipios")
fr = fetch_data("SELECT * FROM public.frota_municipios")

# ---------------------------------------------
# FUNÇÃO DE TRATAMENTO (com cache)
# ---------------------------------------------


@st.cache_data
def tratar_bases(fr: pd.DataFrame, ac: pd.DataFrame):

    # Agrupamento de Frotas
    cols_group = ['codigo_ibge', 'ano', 'municipio']

    frotas_agg = fr.groupby(cols_group).agg({
        'total': 'sum',
        'motocicleta': 'sum',
        'motoneta': 'sum',
        'automovel': 'sum',
        'bonde': 'sum',
        'caminhao': 'sum',
        'caminhao_trator': 'sum',
        'caminhonete': 'sum',
        'camioneta': 'sum',
        'chassi_plataf': 'sum',
        'ciclomotor': 'sum',
        'micro_onibus': 'sum',
        'onibus': 'sum',
        'quadriciclo': 'sum',
        'reboque': 'sum',
        'semi_reboque': 'sum',
        'side_car': 'sum',
        'outros': 'sum',
        'trator_estei': 'sum',
        'trator_rodas': 'sum',
        'triciclo': 'sum',
        'utilitario': 'sum'
    }).reset_index()

    # Tratamento da base de acidentes
    acidentes_renomeado = ac.rename(columns={'ano_acidente': 'ano'})

    acidentes_agg = acidentes_renomeado.groupby(
        ['codigo_ibge', 'ano', 'municipio']
    ).agg({
        'qtde_acidente': 'sum',
        'qtde_acid_com_obitos': 'sum'
    }).reset_index()

    # Ajuste de tipo para merge
    acidentes_agg['codigo_ibge'] = acidentes_agg['codigo_ibge'].astype(str)
    frotas_agg['codigo_ibge'] = frotas_agg['codigo_ibge'].astype(str)

    # Merge final
    df_merged = pd.merge(
        frotas_agg,
        acidentes_agg,
        on=['codigo_ibge', 'ano'],
        how='inner'
    )

    # ---------------------------------------------
    # CÁLCULO DA COLUNA perc_motos
    # ---------------------------------------------

    # garantir que colunas são numéricas
    for col in ['motocicleta', 'motoneta', 'total']:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

    # criar a coluna com proteção contra divisão por zero
    df_merged['perc_motos'] = np.where(
        (df_merged['total'].isna()) | (df_merged['total'] == 0),
        np.nan,
        (df_merged['motocicleta'].fillna(0) +
         df_merged['motoneta'].fillna(0)) / df_merged['total']
    )

    return frotas_agg, acidentes_agg, df_merged


# ---------------------------------------------
# EXECUTAR TRATAMENTO
# ---------------------------------------------
frotas_agg, acidentes_agg, df = tratar_bases(
    fr, ac)  # chamando com a ordem correta

# ==============================
# Separar Brasília e demais municípios
# ==============================
df_brasilia = df[df["municipio_x"] == "BRASILIA"]
df_outros = df[df["municipio_x"] != "BRASILIA"]


# Normalizando os dados
variaveis = ['total', 'automovel',
             'bonde', 'caminhao', 'caminhao_trator', 'caminhonete', 'camioneta',
             'chassi_plataf', 'ciclomotor', 'micro_onibus', 'motocicleta',
             'motoneta', 'onibus', 'quadriciclo', 'reboque', 'semi_reboque',
             'side_car', 'outros', 'trator_estei', 'trator_rodas', 'triciclo',
             'utilitario', 'qtde_acidente', 'qtde_acid_com_obitos',
             'perc_motos']
X = df[variaveis]

scaler = StandardScaler()
X_pad = scaler.fit_transform(X)

# --------------------------------------------------------
# FUNÇÃO CACHEADA PARA ANÁLISE FATORIAL
# --------------------------------------------------------


@st.cache_data
def executar_analise_fatorial(df, n_fatores=5):
    variaveis_af = [
        'automovel', 'caminhao', 'caminhao_trator', 'caminhonete', 'camioneta',
        'chassi_plataf', 'ciclomotor', 'micro_onibus', 'motocicleta',
        'motoneta', 'onibus', 'quadriciclo', 'reboque', 'semi_reboque',
        'side_car', 'outros', 'trator_estei', 'trator_rodas', 'triciclo',
        'utilitario', 'qtde_acidente', 'qtde_acid_com_obitos', 'perc_motos'
    ]
    X_af = df[variaveis_af]

    # Ajuste da análise fatorial
    fa = FactorAnalyzer(n_factors=n_fatores, rotation='varimax')
    fa.fit(X_af)

    # Cargas fatoriais
    cargas = pd.DataFrame(
        fa.loadings_,
        index=X_af.columns,
        columns=[f'Fator{i+1}' for i in range(n_fatores)]
    )

    # Variância explicada
    ss, prop, cum = fa.get_factor_variance()
    variancia = pd.DataFrame(
        [ss, prop, cum],
        index=['SS Loadings', 'Proporção Var.', 'Acumulada'],
        columns=[f'Fator{i+1}' for i in range(n_fatores)]
    )

    return cargas, variancia


# ==============================
# EXECUTAR CCA SIMPLIFICADO
# ==============================
@st.cache_data
def executar_cca_simples(df_BR, df_OUT):
    vars_frota = [
        'automovel', 'caminhao', 'caminhao_trator', 'caminhonete', 'camioneta',
        'chassi_plataf', 'ciclomotor', 'micro_onibus', 'motocicleta',
        'motoneta', 'onibus', 'quadriciclo', 'reboque', 'semi_reboque',
        'side_car', 'outros', 'trator_estei', 'trator_rodas', 'triciclo',
        'utilitario', 'perc_motos'
    ]
    vars_acidente = ['qtde_acidente', 'qtde_acid_com_obitos']

    # Função interna para rodar CCA
    def cca_rodar(df_sub):
        X = df_sub[vars_frota]
        Y = df_sub[vars_acidente]

        # Padronização
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        X_sc = scaler_X.fit_transform(X)
        Y_sc = scaler_Y.fit_transform(Y)

        # Rodar CCA
        cca = CCA(n_components=2)
        cca.fit(X_sc, Y_sc)
        X_c, Y_c = cca.transform(X_sc, Y_sc)

        # Correlações canônicas
        canonical_corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[
            0, 1] for i in range(X_c.shape[1])]

        # Loadings: correlação das variáveis originais com as variáveis canônicas
        load_X = np.corrcoef(X_sc.T, X_c.T)[0:X_sc.shape[1], X_sc.shape[1]:]
        load_Y = np.corrcoef(Y_sc.T, Y_c.T)[0:Y_sc.shape[1], Y_sc.shape[1]:]

        # Cross-loadings: correlação X com Y_c e Y com X_c
        cross_XY = np.corrcoef(X_sc.T, Y_c.T)[0:X_sc.shape[1], X_sc.shape[1]:]
        cross_YX = np.corrcoef(Y_sc.T, X_c.T)[0:Y_sc.shape[1], Y_sc.shape[1]:]

        return {
            'cor': canonical_corrs,
            'X_c': X_c,
            'Y_c': Y_c,
            'load_X': pd.DataFrame(load_X, index=vars_frota, columns=['F1', 'F2']),
            'load_Y': pd.DataFrame(load_Y, index=vars_acidente, columns=['F1', 'F2']),
            'cross_XY': pd.DataFrame(cross_XY, index=vars_frota, columns=['F1', 'F2']),
            'cross_YX': pd.DataFrame(cross_YX, index=vars_acidente, columns=['F1', 'F2'])
        }

    resultados = {
        'Brasília': cca_rodar(df_BR),
        'Outras': cca_rodar(df_OUT)
    }

    return resultados

# ------------------------------------------------------------------


# ---------------------------
# Título do Dashboard
# ---------------------------
st.markdown("<h2>Hipótese 5: Investigação entre o Crescimento da Frota Veicular<br>e a Ocorrência de Acidentes de Trânsito</h2>", unsafe_allow_html=True)

st.markdown("""
Esta análise investiga a relação entre o crescimento da frota de veículos e a ocorrência de acidentes de trânsito nos municípios da RIDE-DF. Utilizando dados de frota veicular e registros de acidentes, buscamos compreender como o aumento no número de veículos impacta a segurança viária. Através de técnicas multivariadas, como Análise Fatorial e Análise de Correlação Canônica (CCA), exploramos padrões e correlações entre o crescimento da frota e a frequência de acidentes, visando fornecer insights valiosos para políticas públicas de trânsito e planejamento urbano.
""")

st.markdown("<br>", unsafe_allow_html=True)


st.markdown("<h4>Visualização da Frota e Acidentes por Município</h4>", unsafe_allow_html=True)

with st.container(border=True):
    # ---------------------------
    # Filtro de intervalo de anos
    # ---------------------------
    ano_min = int(df['ano'].min())
    ano_max = int(df['ano'].max())

    ano_selecionado = st.slider(
        "Selecione o intervalo de anos:",
        min_value=ano_min,
        max_value=ano_max,
        value=(ano_min, ano_max)
    )

    # Filtrar o dataframe
    df_filtrado = df[(df['ano'] >= ano_selecionado[0]) &
                    (df['ano'] <= ano_selecionado[1])]


    # ---------------------------
    # Filtro de municípios
    # ---------------------------
    municipios_disponiveis = df['municipio_x'].unique()
    municipios_selecionados = st.multiselect(
        "Selecione os municípios",
        options=municipios_disponiveis,
        # pode colocar os primeiros 5 como default
        default=list(municipios_disponiveis[:5])
    )

    # Filtrar df pelo que o usuário selecionou
    df_filtrado = df[df['municipio_x'].isin(municipios_selecionados)]

    # ---------------------------
    # Preparar dados para barras triplas
    # ---------------------------
    df_melt = df_filtrado.melt(
        id_vars=['municipio_x'],  # <---- CORRIGIDO
        value_vars=['total', 'qtde_acidente', 'qtde_acid_com_obitos'],
        var_name='Tipo',
        value_name='Quantidade'
    )

    # ---------------------------
    # Gráfico de barras triplas
    # ---------------------------
    fig = px.bar(
        df_melt,
        x='municipio_x',  # <---- CORRIGIDO
        y='Quantidade',
        color='Tipo',
        barmode='group',
        labels={'municipio_x': 'Município',
                'Quantidade': 'Quantidade', 'Tipo': 'Categoria'},
        color_discrete_map={
            'total': '#1f77b4',              # azul
            'qtde_acidente': '#ff7f0e',      # laranja
            'qtde_acid_com_obitos': '#d62728'  # vermelho
        }
    )

    fig.update_layout(
        template='simple_white',
        title='Frota e Acidentes por Município',
        xaxis_tickangle=-45,
        yaxis_title='Quantidade',
        legend_title_text='Categoria',
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    st.plotly_chart(fig, use_container_width=True)


st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h4>Análise Fatorial</h4>", unsafe_allow_html=True)
st.write("Explore os fatores que sintetizam a frota e acidentes por município.")

# ===========================================
# EXECUTAR ANÁLISE FATORIAL COM 5 FATORES
# ===========================================
cargas, variancia = executar_analise_fatorial(df, n_fatores=5)

# ===========================================
# HEATMAP DAS CARGAS FATORIAIS
# ===========================================
with st.container(border=True):
    st.markdown("<h6>Heatmap das Cargas Fatoriais</h6>", unsafe_allow_html=True)
    fig = px.imshow(
        cargas.T,  # Transpose the DataFrame to switch axes
        text_auto=".2f",
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        labels=dict(color="Carga Fatorial")
    )
    fig.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0),
        template="simple_white",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig, use_container_width=False)


# ===========================================
# GRÁFICO DA VARIÂNCIA EXPLICADA
# ===========================================

with st.container(border=True):
    st.markdown("<h6>Variância Explicada pelos Fatores</h6>", unsafe_allow_html=True)
    variancia_t = variancia.T.reset_index().rename(columns={'index': 'Fator'})
    fig_var = px.bar(
        variancia_t,
        x='Fator',
        y='Acumulada',
        text='Acumulada',
        labels={'Acumulada': 'Variância Acumulada'},
        color='Fator'
    )
    fig_var.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_var.update_layout(yaxis=dict(range=[0, 1]), margin=dict(l=50, r=50, t=50, b=50))
    st.plotly_chart(fig_var, use_container_width=True)

    st.markdown("""
            Os 3 primeiros fatores explicam sozinhos quase 88.2% da variância dos dados.
            Isso é excelente, significa que pode-se resumir essas 25 variáveis em apenas 3 ou 4 grupos conceituais (Fatores) sem perder quase nenhuma informação. 
            O fator 4 ainda é relevante (explica 7%), totalizando 95% da variância acumulada. Já o fator 5 representa 1% sendo um ruído plausível de ser descartado.""")
    

# ===========================================
# TABELA COM CARGAS
# ===========================================
with st.container(border=True):
    st.markdown("<h6>Tabela de Cargas Fatoriais</h6>", unsafe_allow_html=True)
    st.dataframe(cargas.style.format("{:.2f}"))
    st.markdown("""- **Fator 1: "Infraestrutura Urbana"**:
    Possui variáveis fortes: trator_rodas (0.79), trator_estei (0.78), qtde_acid_com_obitos (0.79), automovel (0.76), onibus (0.75), side_car (0.77).
    Este fator não é apenas "volume", ele captura a complexidade e o risco.
    Locais com pontuação alta aqui são cidades com muita construção/indústria (tratores), trânsito pesado estabelecido (ônibus/carros) e alta letalidade no trânsito. É o perfil de cidades desenvolvidas ou pólos industriais/agrícolas densos.


- **Fator 2: "Logística Rodoviária e Transporte Econômico"**:
    Possui as seguintes variáveis fortes: ciclomotor (0.80), semi_reboque (0.78), caminhao_trator (0.72), motoneta (0.68).
    Este agrupamento reúne veículos pesados como (semi-reboques/caminhões tratores) juntamente a motonetas e ciclomotes. Isso traz a ideia de zonas mais logísticas, onde possui quantias consideráveis de veículos de carga e ao mesmo tempo de veículos de transporte pessoal mais acessíveis, remetendo a cidades menores e interiores. Os Tratores e Acidentes com Óbito carregam mais forte no fator 1 quando comparado a esse.


- **Fator 3: "Veículos Especiais e Lazer"**:
    Variáveis Fortes: outros (0.71), quadriciclo (0.69), utilitario (0.60), qtde_acidente (0.60).
    Interpretação: Aqui estão os veículos que não são de transporte de massa nem de carga pesada.
    quadriciclo sugere áreas de turismo, praia ou rurais de lazer.
    Curiosamente, "qtde_acidente" carrega bem aqui (0.60), sugerindo que onde há esses veículos misturados, há muitos acidentes, provavelmente não tão letais, já que os acidentes com óbitos ficaram no fator 1.

- O **Fator 4** é dominado somente pera variável perc_motos, medindo o quanto a cidade é dominada por motos. Por ser a única variável percentual, essa "exclusão" faz sentido.

- Já o **Fator 5**, residual, será descartado.
    """)


st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h4>Análise de Correlação Canônica (CCA)</h4>", unsafe_allow_html=True)

# ==============================
# EXECUTAR E EXIBIR NO DASHBOARD
# ==============================
resultados_cca = executar_cca_simples(df_brasilia, df_outros)

st.info("""No output do CCA, obtem-se pares canônicos, que medem scores independentes,
        logo segue-se a análise com foco no F1, de maneira mais objetiva.""", icon=":material/info:""")

with st.container(border=True):
    # Correlação canônica
    st.markdown("<h6>Correlação Canônica</h6>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, border=True, gap="small")
    with col1:
        st.write('Brasília:', resultados_cca['Brasília']['cor'])
    with col2:
        st.write('Outras Cidades:', resultados_cca['Outras']['cor'])

    # Loadings X
    col1, col2 = st.columns(2, border=True, gap="small")
    with col1:
        st.markdown("<h6>Loadings X - Brasília</h6>", unsafe_allow_html=True)
        st.dataframe(resultados_cca['Brasília']['load_X'])
    with col2:
        st.markdown("<h6>Loadings X - Outras Cidades</h6>", unsafe_allow_html=True)
        st.dataframe(resultados_cca['Outras']['load_X'])

    st.info("""Os loadings reconheceram como as 5 variáveis que melhor explicam o próprio grupo de explicativas, em ordem: automovel, caminhao, caminhao trator, caminhonete e camioneta.
            No segundo par F2, teve variações, pois ele detecta variações de cidades, mas não é útil pois as variáveis Y não possuem bom score nesse segundo par.""")


    st.markdown("<br>", unsafe_allow_html=True)
    # Loadings Y
    col1, col2 = st.columns(2, border=True, gap="small")
    with col1:
        st.markdown("<h6>Loadings Y - Brasília</h6>", unsafe_allow_html=True)
        st.dataframe(resultados_cca['Brasília']['load_Y'])
    with col2:
        st.markdown("<h6>Loadings Y - Outras Cidades</h6>", unsafe_allow_html=True)
        st.dataframe(resultados_cca['Outras']['load_Y'])

    st.info("""Diante o grupo das variáveis Y, em ambas a melhor explicativa(próximo a |1|) é a variável quantidade de acidentes, onde os acidentes com óbitos ficam em segundo.
            Porém, nas outras cidades os óbitos possuem um peso maior quando comparados à Brasília em suas cargas canônicas.""")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Cross-loadings X->Y
    col1, col2 = st.columns(2, border=True, gap="small")
    with col1:
        st.markdown("<h6>Cross-loadings X -> Y - Brasília</h6>", unsafe_allow_html=True)
        st.dataframe(resultados_cca['Brasília']['cross_XY'])
    with col2:
        st.markdown("<h6>Cross-loadings X -> Y - Outras Cidades</h6>", unsafe_allow_html=True)
        st.dataframe(resultados_cca['Outras']['cross_XY'])

    st.info("""No caso de Brasília, as variáveis que realmente se destacam na hora de prever a quantidade de acidentes são, em ordem de importância: micro-ônibus, triciclo, automóvel, reboque, semi-reboque, ônibus, caminhão-trator, caminhonete, caminhão, camioneta, utilitário, motocicleta e motoneta. Essas categorias foram as que apresentaram os maiores escores e, por isso, têm maior peso na explicação da variação de acidentes na cidade.
    Já nas outras cidades, o conjunto de variáveis importantes muda um pouco. Aqui, as categorias com maior capacidade de previsão são: ônibus, camioneta, micro-ônibus, automóvel, motocicleta, triciclo, caminhão, reboque e caminhonete. Elas aparecem com os escores mais altos e, portanto, são as que mais contribuem para explicar o número de acidentes nesses municípios.""")

    st.markdown("<br>", unsafe_allow_html=True)

    # Cross-loadings Y->X
    col1, col2 = st.columns(2, border=True, gap="small")
    with col1:
        st.markdown("<h6>Cross-loadings Y -> X - Brasília</h6>", unsafe_allow_html=True)
        st.dataframe(resultados_cca['Brasília']['cross_YX'])
    with col2:
        st.markdown("<h6>Cross-loadings Y -> X - Outras Cidades</h6>", unsafe_allow_html=True)
        st.dataframe(resultados_cca['Outras']['cross_YX'])

    st.info("""Um ponto interessante aparece ao analisar os cross-loadings do Fator 2 (F2). Nele, a variável acidentes com óbitos apresenta um score bem alto e negativo (cerca de –0,9), enquanto o total de acidentes praticamente não se relaciona com esse fator (valor próximo de zero). Isso mostra que o F2 está muito mais associado à gravidade dos acidentes do que à quantidade total de ocorrências. Quando observamos as variáveis preditoras ligadas ao F2 em Brasília, quase nenhuma apresenta força suficiente para ser considerada relevante na predição. Ainda assim, existe uma ligação leve — mais descritiva do que explicativa — com alguns tipos de veículos: quadriciclo (0.63), trator de rodas (0.54) e ciclomotor (0.52). Esses valores indicam apenas uma tendência de associação, mas não configuram variáveis fortes para prever acidentes com óbitos dentro desse fator.""")


    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfico do Par Canônico 1
    st.markdown("<h6>Gráfico Par Canônico 1</h6>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, border=True, gap="small")
    with col1:
        fig_BR = px.scatter(
            x=resultados_cca['Brasília']['X_c'][:, 0],
            y=resultados_cca['Brasília']['Y_c'][:, 0],
            labels={'x': 'Variável Canônica X', 'y': 'Variável Canônica Y'},
            title=f'Brasília - Par Canônico 1 (Correlação: {resultados_cca["Brasília"]["cor"][0]:.3f})'
        )
        st.plotly_chart(fig_BR, use_container_width=True)
    with col2:
        fig_OUT = px.scatter(
            x=resultados_cca['Outras']['X_c'][:, 0],
            y=resultados_cca['Outras']['Y_c'][:, 0],
            labels={'x': 'Variável Canônica X', 'y': 'Variável Canônica Y'},
            title=f'Outras Cidades - Par Canônico 1 (Correlação: {resultados_cca["Outras"]["cor"][0]:.3f})'
        )
        st.plotly_chart(fig_OUT, use_container_width=True)

    st.info("""O scatter reflete em Brasília uma relação perfeita, pontos quase colineares, onde a variabilidade é baixa ou estrutura muito alinhada entre X e Y. Já para outras cidades, percebe-se uma relação forte, mas mais natural, com dispersão, logo maior heterogeneidade nas relações entre frota e acidentes.""")