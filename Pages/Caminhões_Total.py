# Importando bibliotecas necessárias
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
import streamlit as st
#CAMINHÕES TOTAL
# Configuração da página do Streamlit
st.set_page_config(
    page_title='Forecast Licenciamentos Caminhões',
    layout='wide',
    initial_sidebar_state='auto'
)

# Função para carregar dados
def carregar_dados(caminho, index_col):
    try:
        df = pd.read_excel(caminho)
        df[index_col] = pd.to_datetime(df[index_col])
        df.set_index(index_col, inplace=True)
        st.success("Base de dados carregada com sucesso.")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar a base de dados: {e}")
        st.stop()

# Função para configurar e treinar modelo
def configurar_pycaret(df, target_col):
    s = TSForecastingExperiment()
    s.setup(data=df, target=target_col, session_id=123)
    return s

# Layout principal
st.header("Forecast Licenciamentos de Caminhões", divider='green')
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("**Explicação: Licenciamentos Caminhões_Total**", divider='green')
    st.markdown("""
    O segmento de licenciamentos de caminhões é monitorado pela Anfavea, sendo essencial para 
    análise do setor automotivo no Brasil. Utilizamos o PyCaret para previsão com modelos como 
    ARIMA, Prophet e LSTM.
    """)

with col2:
    st.subheader('Imagem Caminhões', divider='green')
    st.image(
        'https://th.bing.com/th/id/OIP.aPSCItq2ardc51c8JfjcXgHaEo?rs=1&pid=ImgDetMain',
        use_container_width=True
    )

# Carregando e exibindo a base de dados
data = carregar_dados(r"C:\Tablets\Caminhão_Total.xlsx", index_col="Mês")

col1, col2, col3 = st.columns([1, 2, 1], gap="large")
with col1:
    st.write("**Base de Dados**")
    st.dataframe(data, use_container_width=True)

# Configuração e treinamento do modelo
modelo = configurar_pycaret(data, target_col="Caminhões")

with col2:
    st.write("**Configuração inicial do PyCaret concluída.**")
    melhor_modelo = modelo.compare_models()
    tabela_comparacao = modelo.pull()
    st.write("### Comparação de Modelos")
    st.dataframe(tabela_comparacao)

# Botão para download da tabela de comparação
    csv_comparacao = tabela_comparacao.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar Comparação", data=csv_comparacao, file_name="comparacao_modelos.csv", mime="text/csv")

with col3:
    with st.expander('***Explicação STLForecaster(sp=12)***'):
        st.markdown('''O STLForecaster é um método de previsão de séries temporais que utiliza a decomposição da série em três componentes principais: Sazonalidade (S), Tendência (T) e Resíduo (L). A decomposição STL, que significa "Seasonal and Trend decomposition using Loess", é amplamente usada por sua flexibilidade e eficácia na modelagem de séries temporais complexas.

Como funciona o STLForecaster?
Decomposição:

A série temporal original é decomposta em Sazonalidade, Tendência e Resíduo.
A componente de sazonalidade captura padrões repetitivos ao longo do tempo (e.g., aumento de vendas no final do ano).
A tendência representa o comportamento de longo prazo da série.
Os resíduos são os valores restantes, após a remoção da sazonalidade e da tendência, capturando variações imprevisíveis.
Modelagem:

Após a decomposição, os resíduos podem ser modelados usando métodos específicos, como ARIMA, Prophet ou outros modelos.
Cada componente pode ser analisado separadamente para ajustar o modelo de forma mais precisa.
Reconstrução e Previsão:

As previsões para cada componente são somadas para reconstruir a série prevista.
Isso garante que padrões sazonais e tendências sejam incorporados corretamente nas previsões.
Por que usar STLForecaster?
Flexibilidade: A decomposição STL permite trabalhar com séries temporais complexas que apresentam sazonalidade variável.
Interpretação clara: Separar os componentes ajuda a entender o que está impulsionando as mudanças na série.
Eficiência: Pode lidar bem com séries ruidosas ou não estacionárias.
Vantagens do STLForecaster
Sazonalidade variável: Funciona bem quando a sazonalidade muda ao longo do tempo.
Robustez a outliers: É resistente a dados discrepantes, graças ao uso de métodos como LOESS (Local Regression).
Integração com outros modelos: Pode ser combinado com métodos mais avançados para modelar os resíduos.
Limitações do STLForecaster
Requer especificação da sazonalidade: É necessário definir a periodicidade da série (e.g., mensal, semanal).
Computacionalmente intensivo: A decomposição STL pode ser mais lenta para séries muito grandes.
Dependência da qualidade da decomposição: O sucesso do modelo depende da separação precisa dos componentes.
Aplicação em Python com PyCaret
No PyCaret, o STLForecaster é usado como parte do módulo time_series, integrando facilmente a decomposição e modelagem.
''')

# Finalização do modelo e previsão
final_model = modelo.finalize_model(melhor_modelo)
st.write("**Modelo Finalizado:**")
st.write(final_model)

col1, col2 = st.columns([1, 1], gap="large")
with col1:
    st.write("**Série Temporal**")
    modelo.plot_model(final_model, plot="ts", display_format="streamlit")

with col2:
    st.write("**Previsão com Horizonte de 36 Períodos**")
    modelo.plot_model(final_model, plot="forecast", data_kwargs={"fh": 36}, display_format="streamlit")

col1, col2=st.columns([1,1], gap='large')

with col1:

# Exportação das previsões
    previsoes = modelo.predict_model(final_model, fh=36)
    st.write("**Previsões:**")
    st.dataframe(previsoes, use_container_width=True)
    csv_previsoes = previsoes.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar Previsões", data=csv_previsoes, file_name="previsoes.csv", mime="text/csv")

with col2:
    st.write('Previsão 2025 e 2026')
    st.metric(label='Previsão 2025', value=119764, delta=192)
    st.metric(label='Previsão 2026', value=119718, delta=-46)