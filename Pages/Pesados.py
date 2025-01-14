# Importando bibliotecas necessárias
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
import streamlit as st
#CAMINHÕES TOTAL
# Configuração da página do Streamlit
st.set_page_config(
    page_title='Forecast Licenciamentos Caminhões Pesados',
    layout='wide',
    initial_sidebar_state='auto'
)

st.header("Forecast Licenciamentos de Caminhões Pesados", divider='blue')

# Layout principal
col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.subheader('Forecast Licenciamentos Caminhões Pesados por meio do PyCaret***', divider='blue')
    st.markdown('''O segmento de licenciamentos de caminhões pesados, monitorado pela Anfavea, inclui veículos com capacidade de carga acima de 24.999 kg. Esses caminhões são essenciais para o transporte de grandes volumes de mercadorias em longas distâncias, oferecendo robustez e eficiência logística. Eles são amplamente utilizados por empresas de logística, construção e indústrias que necessitam de transporte de cargas pesadas e volumosas.
    
O módulo PyCaret Time Series é uma ferramenta avançada para analisar e prever dados de séries temporais usando aprendizado de máquina e técnicas estatísticas clássicas. Esse módulo permite que os usuários executem facilmente tarefas complexas de previsão de séries temporais, automatizando todo o processo, desde a preparação dos dados até a implantação do modelo. 

O módulo PyCaret Time Series Forecasting oferece suporte a uma ampla gama de métodos de previsão, como ARIMA, Prophet e LSTM. Ele também oferece vários recursos para lidar com valores ausentes, decomposição de séries temporais e visualizações de dados. 
        '''
    )

with col2:
    st.subheader('Imagem Caminhão Pesados', divider='blue')
    st.image(
        'https://cdn.autopapo.com.br/box/uploads/2022/01/31183920/mercedes_benz_axor_2544-1536x1024.jpg',
        use_container_width=True
    )

# Carregando o arquivo Excel local
try:
    data = pd.read_excel(r"C:\Tablets\Pesados.xlsx")
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
    s.setup(data=data, target='Pesados', session_id=123)
    st.write("**Configuração inicial do PyCaret concluída.**")

    # Comparar model
    best = s.compare_models()

    # Obter a tabela de comparação
    comparison_df = s.pull()
    st.write("### Comparação de Modelos")
    st.dataframe(comparison_df)

    # Botão para download da tabela de comparação
    csv = comparison_df.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar Comparação", data=csv, file_name="model_comparison.csv", mime='text/csv')

col1, col2, col3  = st.columns([1, 1, 1], gap='large')

with col1:
    st.write('**Time Series - Taget = Caminhões Pesados')
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
       
col1, col2, col3=st.columns([3,1,1], gap='large')
with col1:

    # Realizar previsões
    predictions = s.predict_model(final_best, fh=36)    
    st.write("**Previsões:**")
    st.dataframe(predictions, use_container_width=True)    
# Botão para download da tabela de comparação
    csv = predictions.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar Previsão", data=csv, file_name="predictions.csv", mime='text/csv')

with col2:
    st.write('Previsão 2025')
    st.metric(label='Previsão 2025', value=57572, delta=-4911)

with col3:
    st.write('Previsão 2026')
    st.metric(label='Previsão 2026', value=68198, delta=10626)

col1, col2=st.columns([1,1], gap='large')

with col1:
    with st.expander('Explicação Extra Trees (Extremely Randomized Trees)'):
        st.markdown('''O modelo **Extra Trees** (Extremely Randomized Trees) aplicado ao **forecast de séries temporais** com métodos de **dessazonalização condicional** e **detrending** é uma abordagem híbrida que combina técnicas estatísticas tradicionais de pré-processamento com aprendizado de máquina. Abaixo, explico cada componente e como eles se integram:
---

### **1. Extra Trees (Extremely Randomized Trees)**
- **Definição**: Um algoritmo baseado em árvores de decisão, semelhante ao Random Forest, mas com maior aleatoriedade na construção de árvores. Ele divide os dados escolhendo aleatoriamente:
  - **Atributos**: Para encontrar o melhor ponto de divisão.
  - **Pontos de divisão**: Em vez de buscar o ponto ideal, ele divide aleatoriamente dentro do conjunto de valores possíveis.

- **Vantagens para Séries Temporais**:
  - Alta robustez para lidar com dados ruidosos.
  - Capacidade de capturar interações complexas entre variáveis explicativas.
  - Não requer supostos de linearidade, sendo adequado para séries temporais não lineares.

---

### **2. Cond. Deseasonalize (Dessazonalização Condicional)**
- **O que é?**: Remover os padrões sazonais (como variações diárias, semanais, mensais, ou anuais) que dependem de condições específicas da série.
- **Aplicação**:
  - Identificar padrões sazonais usando métodos estatísticos como médias móveis, decomposição aditiva/multiplicativa, ou transformadas.
  - Ajustar esses padrões com base em condições, como subgrupos de dados (e.g., sazonalidade pode variar entre feriados e dias úteis).

- **Por que é importante?**: 
  - A dessazonalização ajuda o modelo a focar nas tendências subjacentes e nas variações residuais, tornando-o mais eficiente.

---

### **3. Detrending (Remoção de Tendência)**
- **O que é?**: Subtrair ou ajustar tendências globais de longo prazo presentes nos dados, como crescimento ou declínio constante ao longo do tempo.
- **Como é feito?**:
  - Ajustando uma curva de tendência (e.g., regressão linear ou polinomial).
  - Removendo o componente de tendência identificado, deixando os dados estacionários.

- **Benefício**:
  - Simplifica o problema de previsão, permitindo que o modelo Extra Trees foque apenas nas variações residuais e padrões locais.

---

### **4. Aplicação ao Forecast**
Ao integrar essas etapas, a abordagem funciona da seguinte maneira:

1. **Pré-processamento dos Dados**:
   - Remoção de componentes sazonais e tendências para obter uma série residual mais simples e estacionária.

2. **Treinamento do Extra Trees**:
   - O modelo é treinado sobre as características residuais e quaisquer variáveis explicativas criadas (como lags, variáveis sazonais categóricas, etc.).
   - A aleatoriedade inerente ao Extra Trees ajuda a capturar variações complexas e ruidosas.

3. **Reversão do Pré-processamento**:
   - Após realizar as previsões, os componentes removidos (sazonalidade e tendência) são reincorporados às previsões para obter os valores finais.

---

### **Vantagens da Abordagem**
1. **Capacidade de Modelar Relações Não Lineares**:
   - O Extra Trees captura padrões complexos sem a necessidade de explicitá-los no modelo.

2. **Redução de Overfitting**:
   - A aleatoriedade do Extra Trees e o pré-processamento limitam a possibilidade de o modelo aprender padrões específicos de ruído.

3. **Flexibilidade**:
   - O método pode ser aplicado a séries temporais de diferentes tipos, desde que os padrões sazonais e de tendência possam ser identificados.

---

### **Desafios**
1. **Dependência da Qualidade do Pré-processamento**:
   - Se os componentes sazonais e de tendência não forem adequadamente removidos, o desempenho pode ser prejudicado.

2. **Necessidade de Engenharia de Recursos**:
   - Para capturar relações temporais, é necessário criar features como lags, médias móveis, indicadores sazonais, entre outros.

3. **Não Intrinsecamente Temporal**:
   - O Extra Trees não utiliza informações explícitas de sequências temporais (como memória). Para melhorar, pode ser combinado com outras técnicas (e.g., ARIMA para resíduos).

---

### **Resumo**
A abordagem **Extra Trees w/ Cond. Deseasonalize & Detrending** para previsão de séries temporais combina robustez e flexibilidade. Ela tira proveito de métodos estatísticos para simplificar a série temporal e aproveita o poder do Extra Trees para modelar padrões residuais complexos. É especialmente útil em cenários onde os dados têm alta variabilidade e padrões não lineares.
''', unsafe_allow_html=True)
   
with col2:
    with st.expander('**Análise dos Resultados**'):
        st.markdown('''### **Análise dos Resultados: Extra Trees com Cond. Deseasonalize & Detrending**

Os resultados apresentados referem-se ao desempenho de um modelo **Extra Trees** aplicado a séries temporais com dessazonalização condicional e detrending (et_cds_dt). Vamos analisar as métricas:

---

### **1. Desempenho Geral**
- **MASE (0.1751)** e **RMSSE (0.1433)**:
  - Ambos são significativamente baixos, indicando um desempenho muito bom em relação a um modelo de referência (como previsão ingênua).
  - O **RMSSE** mais baixo que o **MASE** sugere que grandes erros estão sendo penalizados de forma eficiente, mas ainda são raros.

  **Interpretação**: O modelo tem uma excelente capacidade de previsão, superando amplamente benchmarks simples.

---

### **2. Erros Absolutos e Quadráticos**
- **MAE (204.1391)** e **RMSE (204.1391)**:
  - Os valores absolutos de erro são relativamente altos, mas são consistentes, como mostra a igualdade entre MAE e RMSE. Isso indica que os erros não estão dispersos ou concentrados em grandes outliers.

  **Interpretação**: Embora os valores sejam elevados em termos absolutos, o erro é uniforme e controlado.

---

### **3. Erros Percentuais**
- **MAPE (3.45%)** e **SMAPE (3.52%)**:
  - Os erros percentuais são extremamente baixos, indicando uma precisão excepcional. Prever com menos de 3.5% de erro médio em séries temporais é uma performance de alto nível.

  **Interpretação**: O modelo consegue capturar a variação da série com grande precisão, mesmo considerando possíveis sazonalidades e tendências.

---

### **4. Tempo de Treinamento**
- **TT (0.0767 segundos)**:
  - O tempo de treinamento é um pouco maior do que os modelos comparados anteriormente, mas ainda extremamente rápido, considerando o uso de um ensemble como Extra Trees.

  **Interpretação**: O desempenho computacional é eficiente, tornando o modelo prático para aplicações em tempo real ou experimentações rápidas.

---

''')