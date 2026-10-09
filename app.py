import streamlit as st
import pandas as pd
import joblib

# 1. Configuração da Página
st.set_page_config(page_title="Fila Inteligente | Central de Suporte", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em; }
    .block-container { max-width: 1500px; padding-top: 2.5rem; padding-bottom: 2rem; }
    [data-testid="stMetric"] { background: linear-gradient(135deg, #ffffff 0%, #f3f7f8 100%); border: 1px solid #dce8e9; border-radius: 12px; padding: 1rem 1.1rem; box-shadow: 0 4px 16px rgba(19, 52, 59, 0.06); }
    [data-testid="stMetricLabel"] { color: #557174; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #13343b; }
    .hero { background: linear-gradient(115deg, #13343b 0%, #1d5960 58%, #2e7770 100%); border-radius: 16px; padding: 1.75rem 2rem; color: #f7fbfa; margin-bottom: 1.5rem; box-shadow: 0 10px 26px rgba(19, 52, 59, 0.18); }
    .hero h1 { color: #ffffff; margin: 0 0 0.35rem 0; font-size: 2.2rem; }
    .hero p { color: #d7ece9; margin: 0; font-size: 1.02rem; }
    .section-label { color: #2e7770; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.75rem; margin-bottom: 0.35rem; }
    .queue-card { border-top: 3px solid #2e7770; padding-top: 0.4rem; }
    .queue-card-orange { border-top-color: #d47b45; }
    .small-note { color: #557174; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
        <div class="section-label" style="color:#b9e2d8">CENTRAL DE OPERAÇÕES · MODELO PREDITIVO</div>
        <h1>Fila Inteligente</h1>
        <p>Priorize chamados pelo impacto financeiro e reduza o risco de cancelamento antes que ele vire perda.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. Carregar Dados e Modelo
@st.cache_data
def carregar_dados():
    df = pd.read_excel("tabela_final_churn.xlsx")
    fila_atual = df.sample(10, random_state=42).copy()
    return fila_atual

@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_gb.pkl")

df_fila = carregar_dados()
modelo = carregar_modelo()

# 3. Preparar os dados para a IA
features_cols = ['Prioridade_Num', 'Customer Age', 'Valor_LTV', 'Ticket Type', 'Ticket Channel']
X_novo = df_fila[features_cols]
X_encoded = pd.get_dummies(X_novo, drop_first=True)
X_encoded = X_encoded.reindex(columns=modelo.feature_names_in_, fill_value=0)

df_fila['Probabilidade_Churn'] = modelo.predict_proba(X_encoded)[:, 1]
df_fila['Risco_IA'] = modelo.predict(X_encoded)
df_fila['Risco (%)'] = df_fila['Probabilidade_Churn'] * 100
df_fila['Faixa de Risco'] = pd.cut(df_fila['Risco (%)'], bins=[-1, 35, 65, 100], labels=['Baixo', 'Atenção', 'Crítico'])

traducao_prioridade = {'High': 'Alta', 'Medium': 'Média', 'Low': 'Baixa'}
traducao_tipo = {
    'Refund request': 'Solicitação de reembolso',
    'Billing inquiry': 'Dúvida de cobrança',
    'Product inquiry': 'Dúvida sobre produto',
}

with st.sidebar:
    st.markdown("### Painel de controle")
    st.caption("Amostra atual da fila de atendimento")
    risco_selecionado = st.multiselect("Exibir faixas de risco", options=['Baixo', 'Atenção', 'Crítico'], default=['Baixo', 'Atenção', 'Crítico'])
    st.divider()
    st.markdown("**Como ler o painel**")
    st.caption("A fila inteligente combina probabilidade de cancelamento e valor do cliente para orientar a ordem de atendimento.")

df_visivel = df_fila[df_fila['Faixa de Risco'].isin(risco_selecionado)].copy()
receita_em_risco = df_fila.loc[df_fila['Risco_IA'] == 1, 'Valor_LTV'].sum()
risco_medio = df_fila['Risco (%)'].mean()
chamados_criticos = (df_fila['Faixa de Risco'] == 'Crítico').sum()

st.markdown('<div class="section-label">Visão executiva</div>', unsafe_allow_html=True)
metricas = st.columns(4)
metricas[0].metric("Chamados analisados", len(df_fila), "fila atual")
metricas[1].metric("Risco médio de cancelamento", f"{risco_medio:.1f}%", "probabilidade média")
metricas[2].metric("Chamados críticos", chamados_criticos, "ação prioritária")
ltv_em_risco_formatado = f"R$ {receita_em_risco:,.0f}".replace(",", ".")
metricas[3].metric("Valor do cliente em risco", ltv_em_risco_formatado, "receita potencial")

grafico_col, leitura_col = st.columns([1.35, 1], gap="large")
with grafico_col:
    st.markdown('<div class="section-label">Mapa de risco</div>', unsafe_allow_html=True)
    risco_por_tipo = df_visivel.groupby('Ticket Type', observed=True)['Risco (%)'].mean().sort_values(ascending=True).to_frame()
    risco_por_tipo.index = risco_por_tipo.index.map(lambda tipo: traducao_tipo.get(tipo, tipo))
    st.bar_chart(risco_por_tipo, horizontal=True, x_label="Risco médio de cancelamento (%)", y_label="Tipo de solicitação", color="#2e7770")
with leitura_col:
    st.markdown('<div class="section-label">Distribuição da fila</div>', unsafe_allow_html=True)
    distribuicao = df_visivel['Faixa de Risco'].value_counts().reindex(['Crítico', 'Atenção', 'Baixo']).fillna(0)
    st.bar_chart(distribuicao, height=245, color="#d47b45")
    st.markdown('<p class="small-note">Chamados críticos devem ser tratados primeiro quando o objetivo é proteger receita.</p>', unsafe_allow_html=True)

st.divider()
st.markdown('<div class="section-label">Comparação operacional</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="queue-card queue-card-orange"><h2>Fila tradicional</h2></div>', unsafe_allow_html=True)
    st.markdown('<p class="small-note">Ordem de chegada: simples, mas sem visibilidade do risco de cancelamento.</p>', unsafe_allow_html=True)
    
    # REMOVIDA A COLUNA DE RISCO DA FILA TRADICIONAL
    df_fifo = df_visivel[['Ticket ID', 'Customer Name', 'Valor_LTV', 'Ticket Priority']].copy()
    df_fifo['Ticket Priority'] = df_fifo['Ticket Priority'].map(traducao_prioridade).fillna(df_fifo['Ticket Priority'])
    df_fifo = df_fifo.rename(columns={'Ticket ID': 'ID do chamado', 'Customer Name': 'Nome do cliente', 'Valor_LTV': 'Valor do cliente (LTV)', 'Ticket Priority': 'Prioridade'})
    st.dataframe(df_fifo, use_container_width=True, hide_index=True, height=355, column_config={'Valor do cliente (LTV)': st.column_config.NumberColumn('Valor do cliente (LTV)', format='R$ %.2f')})

with col2:
    st.markdown('<div class="queue-card"><h2>Fila inteligente</h2></div>', unsafe_allow_html=True)
    st.markdown('<p class="small-note">Ordenada por probabilidade de cancelamento e valor financeiro protegido.</p>', unsafe_allow_html=True)
    df_ia = df_visivel.sort_values(by=['Probabilidade_Churn', 'Valor_LTV'], ascending=[False, False])
    df_ia_display = df_ia[['Ticket ID', 'Customer Name', 'Ticket Type', 'Ticket Priority', 'Valor_LTV', 'Risco (%)', 'Faixa de Risco']].copy()
    df_ia_display['Ticket Type'] = df_ia_display['Ticket Type'].map(traducao_tipo).fillna(df_ia_display['Ticket Type'])
    df_ia_display['Ticket Priority'] = df_ia_display['Ticket Priority'].map(traducao_prioridade).fillna(df_ia_display['Ticket Priority'])
    df_ia_display = df_ia_display.rename(columns={'Ticket ID': 'ID do chamado', 'Customer Name': 'Nome do cliente', 'Ticket Type': 'Tipo de solicitação', 'Valor_LTV': 'Valor do cliente (LTV)', 'Ticket Priority': 'Prioridade'})
    st.dataframe(
        df_ia_display,
        use_container_width=True,
        hide_index=True,
        height=355,
        column_config={
            'Risco (%)': st.column_config.ProgressColumn(
                "Risco de cancelamento (%)",
                help="Probabilidade de cancelamento calculada pela IA",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            'Valor do cliente (LTV)': st.column_config.NumberColumn('Valor do cliente (LTV)', format='R$ %.2f'),
        },
    )

st.divider()
rodape_col1, rodape_col2 = st.columns([2, 1])
with rodape_col1:
    st.markdown("**Fila Inteligente** · Demonstração ao vivo do minicurso FATEC/UNIFIO")
with rodape_col2:
    st.caption("Desenvolvido por Thais Garcia e Isaac")