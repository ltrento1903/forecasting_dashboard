# Importando bibliotecas necessárias
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
import streamlit as st

# Configuração da página do Streamlit
st.set_page_config(
    page_title='Forecast Licenciamentos Caminhões SemiLeves',
    layout='wide',
    initial_sidebar_state='auto'
)

# Cabeçalho da página
st.header("Forecast Licenciamentos de Caminhões SemiLeves", divider='violet')

# Layout principal
col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.subheader('Forecast Licenciamentos Caminhões SemiLeves por meio do PyCaret', divider='violet')
    st.markdown('''O segmento de licenciamentos de caminhões semileves, monitorado pela Anfavea, inclui veículos com capacidade de carga entre 2,5 e 3,5 toneladas. Esses caminhões são essenciais para o transporte de mercadorias em áreas urbanas e rurais, oferecendo flexibilidade e eficiência logística.
    
O módulo PyCaret Time Series é uma ferramenta avançada para analisar e prever dados de séries temporais usando aprendizado de máquina e técnicas estatísticas clássicas. Esse módulo permite que os usuários executem facilmente tarefas complexas de previsão de séries temporais, automatizando todo o processo, desde a preparação dos dados até a implantação do modelo.

O PyCaret Time Series Forecasting oferece suporte a métodos como ARIMA, Prophet e LSTM, além de ferramentas para lidar com valores ausentes, decomposição de séries temporais e visualizações de dados.
    ''')
    
with col2:
    st.subheader('Imagem Caminhão SemiLeves', divider='violet')
    st.image(
        'https://th.bing.com/th/id/OIP.JZRD47hZQ1rj9yXr435AXAHaEK?rs=1&pid=ImgDetMain',
        use_container_width=True
    )

# Carregando o arquivo Excel local
try:
    data = pd.read_excel(r"C:\Tablets\Semileves.xlsx")
    data['Mês'] = pd.to_datetime(data['Mês'])  # Ajustar o nome da coluna de data, se necessário
    data.set_index('Mês', inplace=True)   # Definir a coluna de data como índice
    st.success("Base de dados carregada com sucesso.")         
except Exception as e:
    st.error(f"Erro ao carregar a base de dados: {e}")
    st.stop()

# Visualizar a base de dados no Streamlit
col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.write('***Base de Dados Anfavea***')
    st.dataframe(data, use_container_width=True)    

with col2:
    # Configuração inicial do experimento
    s = TSForecastingExperiment()
    s.setup(data=data, target='Semileves', session_id=123)
    st.write("**Configuração inicial do PyCaret concluída.**")

    # Comparar modelos
    best = s.compare_models()

    # Obter a tabela de comparação
    comparison_df_sl = s.pull()
    st.write("### Comparação de Modelos")
    st.dataframe(comparison_df_sl)

    # Botão para download da tabela de comparação
    csv_me = comparison_df_sl.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Baixar Comparação",
        data=csv_me,
        file_name="model_comparison.csv",
        mime="text/csv",
        key="download_button_comparison_sl"
    )

# Seção para visualização e previsões
col1, col2, col3 = st.columns([1, 1, 1], gap='large')

with col1:
    st.write('**Time Series - Target = Caminhões Semileves**')
    s.plot_model(best, plot='ts', display_format='streamlit')  

with col2:
    # Finalizar o modelo
    final_best = s.finalize_model(best)
    st.write("**Modelo finalizado:**")
    st.write(final_best)

with col3:
    # Plotar previsões
    st.write("**Previsão com horizonte de 36 períodos:**")
    s.plot_model(final_best, plot='forecast', data_kwargs={'fh': 36}, display_format='streamlit')

# Exibindo previsões e métricas
col1, col2, col3 = st.columns([3, 1, 1], gap='large')

with col1:
    predictions = s.predict_model(final_best, fh=36)       
    st.write("**Previsões:**")
    st.dataframe(predictions, use_container_width=True)
    
    # Botão para download das previsões
    csv = predictions.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar Previsão",
                        data=csv, 
                        file_name="predictions.csv",
                        mime='text/csv',
                        key='download_button_previsao_sl')

with col2:
    st.write('Previsão 2025')
    st.metric(label='Previsão 2025', value=734, delta=-291)

with col3:
    st.write('Previsão 2026')
    st.metric(label='Previsão 2026', value=547, delta=-187)


col1, col2=st.columns([1,1], gap='large')

with col1:
    with st.expander('Explicação AdaBoost w/ Cond. Deseasonalize & Detrending'):
        st.markdown('''O **AdaBoostRegressor** (Adaptive Boosting Regressor) é um algoritmo de aprendizado de máquina baseado em ensemble que combina múltiplos estimadores fracos (geralmente árvores de decisão) para construir um modelo forte e robusto. Quando aplicado a problemas de previsão de séries temporais, ele precisa ser ajustado de forma específica, pois o AdaBoost não lida diretamente com as dependências temporais intrínsecas desses dados.

### **Como funciona o AdaBoostRegressor no contexto geral**
- **Aprendizado Iterativo**: Ele ajusta um modelo base (como uma árvore de decisão) em várias iterações, dando maior peso às amostras que não foram previstas corretamente em cada etapa.
- **Combinação de Modelos**: Os resultados dos modelos ajustados são combinados para produzir uma previsão final ponderada.
- **Robustez**: Funciona bem para dados complexos e não linearidades.

### **Aplicação em Séries Temporais**
Séries temporais têm uma característica única: a dependência temporal entre observações. O AdaBoost não possui suporte nativo para essas dependências, então é necessário transformar os dados para incorporar essa estrutura antes de aplicar o modelo.

### **Vantagens de usar AdaBoost em séries temporais**
- **Lida bem com não linearidades**: Útil para séries temporais complexas com padrões não lineares.
- **Combinação de modelos fracos**: Pode capturar diferentes aspectos do padrão temporal ao longo das iterações.
- **Flexibilidade**: Permite ajustar o regressor base (ex.: árvores de decisão).

---

### **Desafios**
- **Pré-processamento necessário**: Requer criação de lags e engenharia de atributos, o que pode ser trabalhoso.
- **Não captura dependências longas diretamente**: Pode ser limitado para séries com alta autocorrelação em intervalos muito distantes.
- **Sensível ao ruído**: O AdaBoost pode exagerar a importância de observações ruidosas em séries temporais pequenas.

---

### **Alternativas**
Se a série for altamente dependente do tempo ou sazonal, considere algoritmos projetados para séries temporais, como:
- ARIMA/Prophet (estatísticos).
- Redes Neurais Recorrentes (RNNs), LSTMs.
- Modelos baseados em ensembles com suporte nativo para séries temporais (Ex.: Facebook Prophet).

O **AdaBoostRegressor** é uma escolha viável, mas é mais adequado para séries temporais curtas ou não lineares quando técnicas tradicionais não se adequam bem.
''', unsafe_allow_html=True)
   
with col2:
    with st.expander('**Análise dos Resultados**'):
        st.markdown('''Aqui está uma análise detalhada do desempenho do modelo **AdaBoost com Deseasonalização Condicional e Detrending** (ada_cds_dt), baseado nas métricas apresentadas:

---

### **Métricas Avaliadas**

1. **MASE (Mean Absolute Scaled Error)**  
   - **Valor reportado**: 0.7813  
   - **Interpretação**: O MASE compara o desempenho do modelo com um modelo de referência simples, como um modelo Naïve.  
     - Valor < 1: O modelo supera o Naïve.
     - Valor > 1: O modelo é pior que o Naïve.  
   - **Análise**: Com um MASE de 0.7813, o modelo supera o modelo Naïve. Contudo, seu desempenho está longe de ser considerado excelente.

---

2. **RMSSE (Root Mean Squared Scaled Error)**  
   - **Valor reportado**: 0.5807  
   - **Interpretação**: Métrica semelhante ao MASE, mas penaliza erros maiores com mais intensidade.  
   - **Análise**: O RMSSE indica que o modelo lida razoavelmente bem com erros maiores, mantendo um desempenho relativamente bom em comparação com o modelo de referência.

---

3. **MAE (Mean Absolute Error)**  
   - **Valor reportado**: 27.5790  
   - **Interpretação**: O erro absoluto médio é a média das diferenças absolutas entre os valores previstos e reais.  
   - **Análise**: Um valor de 27.5790 pode ser considerado alto ou baixo dependendo da escala dos dados, mas sem esse contexto específico, é difícil tirar conclusões detalhadas.

---

4. **RMSE (Root Mean Squared Error)**  
   - **Valor reportado**: 27.5790  
   - **Interpretação**: Penaliza erros maiores mais severamente que o MAE.  
   - **Análise**: Como o RMSE e o MAE são iguais aqui, isso sugere que os erros são relativamente consistentes, sem grandes outliers.

---

5. **MAPE (Mean Absolute Percentage Error)**  
   - **Valor reportado**: 0.2316 (23.16%)  
   - **Interpretação**: Representa o erro médio absoluto como uma porcentagem dos valores reais.  
   - **Análise**: Um MAPE de 23.16% é relativamente alto, indicando que o modelo pode não ser ideal para previsões precisas em algumas aplicações sensíveis.

---

6. **SMAPE (Symmetric Mean Absolute Percentage Error)**  
   - **Valor reportado**: 0.2716 (27.16%)  
   - **Interpretação**: Métrica simétrica que evita distorções causadas por valores extremos.  
   - **Análise**: O SMAPE reforça a ideia de que o modelo apresenta erros significativos em relação à magnitude dos valores previstos e reais.

---

7. **TT (Tempo Total)**  
   - **Valor reportado**: 0.0433 segundos  
   - **Interpretação**: Reflete o tempo computacional para treinamento e previsão.  
   - **Análise**: O modelo é extremamente eficiente em termos de tempo, levando apenas 0.0433 segundos.

---

### **Pontos Fortes**
- **Eficiência Computacional**: O modelo é muito rápido (TT = 0.0433).
- **Desempenho Relativo ao Naïve**: Com MASE e RMSSE menores que 1, o modelo supera o baseline.

---

### **Pontos Fracos**
- **Erros Percentuais Altos**: O MAPE (23.16%) e SMAPE (27.16%) indicam que o modelo apresenta dificuldades para capturar a magnitude exata dos valores previstos.
- **Erro Médio Absoluto**: Dependendo da escala dos dados, o MAE de 27.5790 pode ser significativo.

---

### **Conclusão**
O modelo **AdaBoost com Deseasonalização e Detrending** é eficiente em termos de tempo e apresenta desempenho razoável em relação ao modelo Naïve. No entanto, os altos valores de MAPE e SMAPE indicam que ele pode não ser adequado para aplicações onde a precisão é essencial. Melhorias podem ser exploradas ajustando os hiperparâmetros do AdaBoost, refinando o pré-processamento dos dados ou testando alternativas como modelos específicos para séries temporais (ex.: ARIMA, Prophet, LSTMs).
''')