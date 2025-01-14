# Importando bibliotecas necessárias
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
import streamlit as st
#CAMINHÕES TOTAL
# Configuração da página do Streamlit
st.set_page_config(
    page_title='Forecast Licenciamentos Caminhões Médios',
    layout='wide',
    initial_sidebar_state='auto'
)

#Caminhões Médios

st.header("Forecast Licenciamentos de Caminhões Médios", divider='violet')

# Layout principal
col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.subheader('Forecast Licenciamentos Caminhões Médios por meio do PyCaret***', divider='violet')
    st.markdown('''O segmento de licenciamentos de caminhões médios, monitorado pela Anfavea, inclui veículos com capacidade de carga entre 10.000 kg e 14.999 kg. Esses caminhões são essenciais para o transporte de mercadorias em áreas urbanas e rurais, oferecendo flexibilidade e eficiência logística. Eles são amplamente utilizados por empresas de logística, comércio e construção, devido à sua capacidade de transportar cargas moderadas com agilidade e economia.
    
O módulo PyCaret Time Series é uma ferramenta avançada para analisar e prever dados de séries temporais usando aprendizado de máquina e técnicas estatísticas clássicas. Esse módulo permite que os usuários executem facilmente tarefas complexas de previsão de séries temporais, automatizando todo o processo, desde a preparação dos dados até a implantação do modelo. 

O módulo PyCaret Time Series Forecasting oferece suporte a uma ampla gama de métodos de previsão, como ARIMA, Prophet e LSTM. Ele também oferece vários recursos para lidar com valores ausentes, decomposição de séries temporais e visualizações de dados. 
        '''
    )

with col2:
    st.subheader('Imagem Caminhão Médios', divider='violet')
    st.image(
        'https://fotos-estradao-estadao.nyc3.cdn.digitaloceanspaces.com/wp-content/uploads/2024/01/08082121/Accelo-1017.jpg',
        use_container_width=True
    )

# Carregando o arquivo Excel local
try:
    data = pd.read_excel(r"C:\Tablets\Medios.xlsx")
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
    s.setup(data=data, target='Médios', session_id=123)
    st.write("**Configuração inicial do PyCaret concluída.**")

    # Comparar model
    best = s.compare_models()

    # Obter a tabela de comparação
    comparison_df_me = s.pull()
    st.write("### Comparação de Modelos")
    st.dataframe(comparison_df_me)

    # Botão para download da tabela de comparação
    csv_me = comparison_df_me.to_csv(index=False).encode('utf-8')
    st.download_button(
    "Baixar Comparação",
    data=csv_me,
    file_name="model_comparison.csv",
    mime="text/csv",
    key="download_button_comparison_me"  # Chave única
    )

col1, col2, col3  = st.columns([1, 1, 1], gap='large')

with col1:
    st.write('**Time Series - Taget = Caminhões Médios')
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
    st.download_button(
        label="Baixar Previsão",
        data=csv, 
        file_name="predictions.csv",
        mime='text/csv',
        key='download_button_previsao_me'
    )

with col2:
    st.write('Previsão 2025')
    st.metric(label='Previsão 2025', value=10000, delta=648)

with col3:
    st.write('Previsão 2026')
    st.metric(label='Previsão 2026', value=11049, delta=1049)


col1, col2=st.columns([1,1], gap='large')

with col1:
    with st.expander('Explicação Decision Tree w/ Cond. Deseasonalize & Detrending'):
        st.markdown('''### **Decision Tree w/ Cond. Deseasonalize & Detrending em Forecast de Série Temporal**

O modelo **Decision Tree com Condicional Deseasonalize & Detrending** aplicado a **forecast de séries temporais** utiliza uma abordagem híbrida para lidar com características complexas das séries temporais, como sazonalidade e tendência, antes de aplicar o modelo de aprendizado supervisionado.

---

### **1. Componentes da Abordagem**

#### **Decision Tree (Árvore de Decisão)**
As árvores de decisão são modelos de aprendizado supervisionado que particionam os dados com base em condições de divisão. No contexto de séries temporais, essas condições podem ser baseadas em:
- **Lags temporais:** Valores passados da série (ex.: \(y_{t-1}, y_{t-2}, \dots\)).
- **Outras variáveis derivadas:** Tendência, sazonalidade, ou variáveis externas (exógenas).

O modelo busca prever o valor da série (\(y_t\)) para um horizonte de tempo específico, com base nesses inputs.

#### **Condicional Deseasonalize & Detrending**
Antes de treinar o modelo, os componentes de **sazonalidade** e **tendência** são identificados e, se considerados significativos, removidos.  
- **Deseasonalize:** Remove padrões periódicos que se repetem em intervalos regulares (diários, mensais, etc.).
- **Detrending:** Remove o crescimento ou declínio de longo prazo.

Essa remoção é **condicional**, ou seja, só ocorre se:
- A sazonalidade for significativa (identificada por métodos estatísticos ou heurísticos).
- A tendência for detectada como dominante.

---

### **2. Fluxo de Aplicação no Forecast de Série Temporal**

1. **Pré-processamento da Série Temporal**
   - **Identificação da sazonalidade e tendência:** Usar técnicas como decomposição STL ou testes estatísticos para determinar os componentes dominantes da série.
   - **Remoção condicional:** Subtrair ou dividir a série pela tendência e/ou sazonalidade, criando uma série residual estacionária.

2. **Criação de Variáveis Predictoras**
   - **Lags temporais:** Criar colunas como \(y_{t-1}, y_{t-2}, \dots, y_{t-k}\), onde \(k\) é o número de lags usado.
   - **Sazonalidade ou variáveis exógenas:** Adicionar componentes sazonais ou outras variáveis externas, caso presentes.

3. **Treinamento do Modelo**
   - Usar os dados transformados para treinar o modelo **Decision Tree**. Este modelo aprende os padrões residuais após a remoção dos componentes sazonais e de tendência.

4. **Previsão**
   - Realizar previsões para o horizonte desejado (\(fh\)).
   - Reconstruir a série original reintroduzindo os componentes de sazonalidade e tendência removidos durante o pré-processamento.

---

### **3. Vantagens da Abordagem**

1. **Flexibilidade do Modelo**
   - Decision Trees capturam padrões não lineares e interações complexas entre os lags e outras variáveis.

2. **Redução da Complexidade**
   - A remoção de tendência e sazonalidade reduz a complexidade dos dados, permitindo que o modelo foque em padrões residuais.

3. **Facilidade de Interpretação**
   - O modelo Decision Tree é altamente interpretável, especialmente quando comparado a modelos complexos como redes neurais.

4. **Adaptável a Séries Multivariadas**
   - Permite a incorporação de variáveis exógenas (ex.: fatores econômicos, clima).

---

### **5. Limitações**
1. **Perda de Informação**
   - A remoção de componentes pode introduzir vieses ou descartar informações importantes.
2. **Sazonalidade Dinâmica**
   - Caso os padrões sazonais mudem ao longo do tempo, a remoção condicional pode não capturar essas variações.
3. **Overfitting**
   - Decision Trees podem superajustar dados ruidosos, especialmente em séries temporais curtas.

---

### **6. Quando Usar**
- **Séries Temporais com Sazonalidade e Tendência Significativas**
  - Exemplos: vendas mensais, consumo de energia elétrica.
- **Padrões Não Lineares**
  - Quando a relação entre os lags e o valor futuro não é linear.
- **Conjuntos de Dados Pequenos**
  - Decision Trees têm bom desempenho em cenários com menos dados em comparação a modelos mais complexos.

---

### **Conclusão**
A aplicação de **Decision Tree com Condicional Deseasonalize & Detrending** é uma estratégia eficiente para previsões de séries temporais que apresentam sazonalidade e tendência significativas. A remoção prévia desses componentes permite ao modelo capturar padrões residuais de forma mais precisa, tornando-o uma abordagem útil e interpretável para uma ampla gama de problemas de previsão.
''', unsafe_allow_html=True)
   
with col2:
    with st.expander('**Análise dos Resultados**'):
        st.markdown('''### ### Análise do Modelo: Resultados Atualizados

A análise apresentada refere-se a um modelo de Árvore de Decisão utilizando dessazonalização condicional e detrending (dt_cds_dt), avaliado com diversas métricas de erro e tempo de treinamento. Vamos interpretar os resultados:

---

### **1. Desempenho Geral**
- **MASE (0.3916)** e **RMSSE (0.2973)**:
  - Valores baixos indicam que o modelo performa bem em relação a um benchmark simples (como previsão média ou ingênua). 
  - O RMSSE (métrica baseada em quadrados) é menor que o MASE, sugerindo que grandes erros não são frequentes e os desvios são limitados.

  **Interpretação**: O modelo é confiável e apresenta consistência ao superar a abordagem de referência.

---

### **2. Erros Absolutos e Quadráticos**
- **MAE (46.1750)** e **RMSE (46.1750)**:
  - A igualdade entre MAE e RMSE indica que os erros não possuem grandes outliers; os desvios estão bem distribuídos.

  **Interpretação**: Este resultado reforça que o modelo apresenta previsões uniformes, com erros concentrados e sem grandes variações.

---

### **3. Erros Percentuais**
- **MAPE (5.21%)** e **SMAPE (5.40%)**:
  - Esses valores indicam que os erros médios estão abaixo de 6% em relação aos valores reais, com o SMAPE levemente superior devido à sua simetria.

  **Interpretação**: A precisão do modelo é alta, sendo muito eficiente para problemas onde os erros relativos são críticos.

---

### **4. Tempo de Treinamento**
- **TT (0.0667 segundos)**:
  - O modelo treina de forma extremamente rápida, tornando-o ideal para aplicações com restrições de tempo ou iteração frequente.

  **Interpretação**: O tempo de treinamento ligeiramente superior ao exemplo anterior (0.036) ainda é competitivo e adequado para uso prático.

---

### **Comparação com o Exemplo Anterior**
- **MAPE/SMAPE**: Este modelo (5.21%/5.40%) é levemente mais preciso do que o anterior (5.96%/5.97%).
- **MASE/RMSSE**: Ambos os erros escalados também são um pouco melhores neste modelo.
- **MAE/RMSE**: O erro absoluto foi reduzido de ~198 para ~46, indicando previsões mais precisas.
- **TT**: O tempo de treinamento é maior, mas ainda extremamente rápido e negligenciável em comparação ao benefício da precisão.

---

### **Conclusão**
- **Pontos Fortes**:
  - Alta precisão em erros percentuais e absolutos.
  - Baixa penalização de grandes erros (indicada pelo RMSSE).
  - Tempo de treinamento ainda eficiente.
- **Pontos de Melhoria**:
  - A análise não inclui métricas de validação cruzada ou desempenho em dados fora da amostra.
  - Investigar a robustez do modelo em diferentes cenários ou séries temporais com maior variabilidade.

Gostaria de explorar mais comparações ou detalhar estratégias para melhorar ainda mais o desempenho?
''')