import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Dashboard de Análise de Cancelamentos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# FUNÇÕES DE APOIO (CARREGAMENTO, LIMPEZA, MODELAGEM)
# ==============================================================================

# Cache para acelerar o carregamento dos dados
@st.cache_data
def load_data(path):
    """Carrega e limpa os dados do hotel."""
    df = pd.read_csv(path)
    # Copiar para evitar alterar o cache
    df_clean = df.copy()

    # Remover coluna 'company' (muitos valores nulos)
    if 'company' in df_clean.columns:
        df_clean = df_clean.drop('company', axis=1)

    # Preencher valores nulos
    df_clean['agent'].fillna(0, inplace=True)
    if df_clean['country'].isnull().any():
        df_clean['country'].fillna(df_clean['country'].mode()[0], inplace=True)
    if df_clean['children'].isnull().any():
        df_clean['children'].fillna(df_clean['children'].mode()[0], inplace=True)

    # Converter tipos de dados
    df_clean[['children', 'agent']] = df_clean[['children', 'agent']].astype(int)

    # Remover dados inconsistentes
    df_clean = df_clean[~((df_clean['adults'] == 0) & (df_clean['children'] == 0) & (df_clean['babies'] == 0))]
    return df_clean

@st.cache_data
def get_country_mapping():
    """Retorna um dicionário para mapear códigos de países para nomes."""
    # Mapeamento simplificado para os países mais comuns no dataset
    return {
        'PRT': 'Portugal', 'GBR': 'Reino Unido', 'USA': 'Estados Unidos', 'ESP': 'Espanha',
        'IRL': 'Irlanda', 'FRA': 'França', 'ROU': 'Romênia', 'NOR': 'Noruega',
        'OMN': 'Omã', 'ARG': 'Argentina', 'POL': 'Polônia', 'DEU': 'Alemanha',
        'BEL': 'Bélgica', 'CHE': 'Suíça', 'CN': 'China', 'ITA': 'Itália',
        'NLD': 'Holanda', 'DNK': 'Dinamarca', 'SWE': 'Suécia', 'BRA': 'Brasil'
    }

def train_model(df, features, apply_smote):
    """Treina o modelo de regressão logística e retorna o modelo e os dados de teste."""
    X = df[features]
    y = df['is_canceled']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    model = LogisticRegression(solver='liblinear', max_iter=2000)
    model.fit(X_train, y_train)

    return model, X_test, y_test

def interpret_coefficients(model, features):
    """Gera uma interpretação automática dos coeficientes."""
    coef_df = pd.DataFrame({
        'Variável': features,
        'Coeficiente': model.coef_[0]
    })
    coef_df['Odds Ratio'] = np.exp(coef_df['Coeficiente'])
    coef_df['Importância'] = np.abs(coef_df['Coeficiente'])
    coef_df = coef_df.sort_values(by='Importância', ascending=False)

    st.subheader("🤖 Interpretação Automática dos Fatores")
    for i in range(min(5, len(coef_df))):
        row = coef_df.iloc[i]
        var = row['Variável'].replace("_", " ").title()
        odds = row['Odds Ratio']

        if odds > 1:
            st.error(f"**{var}** é um **fator de risco**. Cada unidade de aumento nesta variável aumenta a chance de cancelamento em **{odds:.2f} vezes**.")
        else:
            reduction = (1 - odds) * 100
            st.success(f"**{var}** é um **fator de proteção**. Cada unidade de aumento nesta variável diminui a chance de cancelamento em **{reduction:.1f}%**.")
    return coef_df

# ==============================================================================
# INTERFACE DO STREAMLIT (SIDEBAR)
# ==============================================================================
st.sidebar.header("⚙️ Painel de Controle")

# Carregar dados
try:
    df_full = load_data('hotel_bookings.csv')
    df = df_full.copy() # Criar uma cópia para manipulação
except FileNotFoundError:
    st.error("ERRO: O arquivo 'hotel_bookings.csv' não foi encontrado. Por favor, coloque-o no mesmo diretório do app.")
    st.stop()


# Filtros de dados
st.sidebar.subheader("1. Filtros de Visualização")
hotel_type = st.sidebar.selectbox(
    "🏨 Tipo de Hotel:",
    ("Ambos", "City Hotel", "Resort Hotel"),
    help="Filtre os dados para ver a análise de um tipo de hotel específico ou de ambos."
)
if hotel_type != "Ambos":
    df = df[df['hotel'] == hotel_type]

# Filtro por mês
meses = ['Todos'] + list(df['arrival_date_month'].unique())
selected_month = st.sidebar.selectbox("🗓️ Mês de Chegada:", meses)
if selected_month != 'Todos':
    df = df[df['arrival_date_month'] == selected_month]

# ==============================================================================
# TÍTULO E INTRODUÇÃO
# ==============================================================================
st.title("📊 Dashboard Preditivo de Cancelamento de Reservas")
st.markdown("""
Esta ferramenta interativa utiliza um modelo de Regressão Logística para analisar e prever
a probabilidade de cancelamento de reservas de hotel. Use o painel de controle à esquerda para filtrar
os dados, selecionar variáveis e treinar o modelo.
""")

# ==============================================================================
# ABA 1: ANÁLISE EXPLORATÓRIA
# ==============================================================================
st.header("🌍 Análise Exploratória dos Dados")

# Métricas principais
total_reservas = df.shape[0]
taxa_cancelamento = (df['is_canceled'].sum() / total_reservas) * 100
diaria_media = df['adr'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total de Reservas (Filtrado)", f"{total_reservas:,}")
col2.metric("Taxa de Cancelamento", f"{taxa_cancelamento:.1f}%")
col3.metric("Diária Média (ADR)", f"€ {diaria_media:.2f}")

# Gráficos exploratórios
st.subheader("Visualizações Interativas")
col_a, col_b = st.columns(2)

with col_a:
    # Gráfico de taxa de cancelamento por mês
    df_monthly = df.groupby('arrival_date_month')['is_canceled'].mean().reset_index()
    df_monthly['arrival_date_month'] = pd.Categorical(df_monthly['arrival_date_month'], categories=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], ordered=True)
    df_monthly = df_monthly.sort_values('arrival_date_month')
    fig_monthly = px.bar(df_monthly, x='arrival_date_month', y='is_canceled', title='Taxa Média de Cancelamento por Mês', labels={'is_canceled': 'Taxa de Cancelamento', 'arrival_date_month': 'Mês'})
    st.plotly_chart(fig_monthly)

with col_b:
    # Mapa Mundi
    country_data = df['country'].value_counts().reset_index()
    country_data.columns = ['code', 'count']
    country_map = get_country_mapping()
    country_data['country'] = country_data['code'].map(country_map)
    country_data = country_data.dropna()

    fig_map = px.choropleth(country_data,
                            locations="code",
                            color="count",
                            hover_name="country",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title="Origem das Reservas por País")
    st.plotly_chart(fig_map)


# ==============================================================================
# ABA 2: MODELAGEM PREDITIVA
# ==============================================================================
st.header("🧠 Modelagem Preditiva Interativa")

# Seleção de variáveis
st.sidebar.subheader("2. Configurações do Modelo")
todas_features = ['lead_time', 'is_repeated_guest', 'previous_cancellations', 'required_car_parking_spaces', 'total_of_special_requests', 'adr']
variaveis_selecionadas = st.sidebar.multiselect(
    "📊 Escolha as variáveis para o modelo:",
    options=todas_features,
    default=todas_features,
    help="Selecione as variáveis que o modelo usará para fazer a previsão."
)

# Adicionar dummies do hotel se não estiver filtrado
if hotel_type == "Ambos":
    df['hotel_Resort Hotel'] = (df['hotel'] == 'Resort Hotel').astype(int)
    if 'hotel_Resort Hotel' not in variaveis_selecionadas:
      variaveis_selecionadas.append('hotel_Resort Hotel')


# Opção de balanceamento
aplicar_smote = st.sidebar.checkbox("Balancear dados com SMOTE", value=True, help="Use SMOTE para corrigir o desbalanceamento entre cancelamentos e não-cancelamentos, geralmente melhora a captura de cancelamentos.")

if st.sidebar.button("🚀 Treinar Modelo e Gerar Previsões"):
    if not variaveis_selecionadas:
        st.warning("Por favor, selecione ao menos uma variável para treinar o modelo.")
    else:
        with st.spinner("Treinando modelo... Isso pode levar um momento."):
            # Treinar o modelo
            model, X_test, y_test = train_model(df, variaveis_selecionadas, aplicar_smote)
            st.success("Modelo treinado com sucesso!")

            # Fazer previsões
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Mostrar resultados
            tab1, tab2, tab3 = st.tabs(["📈 Métricas de Performance", "🧠 Interpretação do Modelo", "📉 Curva Logística"])

            with tab1:
                st.subheader("Métricas de Performance")
                # Matriz de Confusão
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, title="Matriz de Confusão",
                                   labels=dict(x="Previsto", y="Real"),
                                   x=['Não Cancela', 'Cancela'], y=['Não Cancela', 'Cancela'],
                                   color_continuous_scale="Greens")
                st.plotly_chart(fig_cm)

                # Curva ROC
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = roc_auc_score(y_test, y_prob)
                fig_roc = go.Figure(data=go.Scatter(x=fpr, y=tpr, name=f'Curva ROC (AUC = {auc_score:.3f})'))
                fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                fig_roc.update_layout(title_text='Curva ROC', xaxis_title='Taxa de Falsos Positivos', yaxis_title='Taxa de Verdadeiros Positivos')
                st.plotly_chart(fig_roc)

                # Relatório de Classificação
                st.text("Relatório de Classificação Detalhado:")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

            with tab2:
                # Interpretação dos coeficientes
                coef_df = interpret_coefficients(model, variaveis_selecionadas)
                st.subheader("Tabela Completa de Coeficientes")
                st.dataframe(coef_df.style.format({
                    'Coeficiente': '{:.3f}',
                    'Odds Ratio': '{:.3f}',
                    'Importância': '{:.3f}'
                }))


            with tab3:
                st.subheader("Curva Logística Interativa")
                feature_curva = st.selectbox("Selecione uma variável para visualizar sua curva logística:", options=variaveis_selecionadas)

                # Gerar a curva
                X_train, _, _, _ = train_test_split(df[variaveis_selecionadas], df['is_canceled'], test_size=0.3, random_state=42)
                grid = np.linspace(X_train[feature_curva].min(), X_train[feature_curva].max(), 100)
                temp_X = pd.DataFrame(np.tile(X_train.mean().values, (100, 1)), columns=variaveis_selecionadas)
                temp_X[feature_curva] = grid
                
                probs = model.predict_proba(temp_X)[:, 1]
                
                fig_log = px.line(x=grid, y=probs, labels={'x': feature_curva, 'y': 'Probabilidade de Cancelamento'})
                fig_log.update_layout(title=f"Probabilidade de Cancelamento vs. '{feature_curva}'")
                st.plotly_chart(fig_log)
else:
    st.info("👈 Use o painel de controle à esquerda para configurar e treinar o modelo preditivo.")
