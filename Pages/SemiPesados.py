# Importando bibliotecas necessárias
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
import streamlit as st
#CAMINHÕES TOTAL
# Configuração da página do Streamlit
st.set_page_config(
    page_title='Forecast Licenciamentos Caminhões Semipesados',
    layout='wide',
    initial_sidebar_state='auto'
)

st.header("Forecast Licenciamentos de Caminhões Semipesados", divider='red')

# Layout principal
col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.subheader('Forecast Licenciamentos Caminhões Semipesados por meio do PyCaret', divider='red')
    st.markdown('''O segmento de licenciamentos de caminhões semipesados, monitorado pela Anfavea, inclui veículos com capacidade de carga entre 15.000 kg e 24.999 kg. Esses caminhões são essenciais para o transporte de mercadorias em áreas urbanas e rurais, oferecendo flexibilidade e eficiência logística. Eles são amplamente utilizados por empresas de logística, comércio e construção, devido à sua capacidade de transportar cargas moderadas com agilidade e economia.
    
O módulo PyCaret Time Series é uma ferramenta avançada para analisar e prever dados de séries temporais usando aprendizado de máquina e técnicas estatísticas clássicas. Esse módulo permite que os usuários executem facilmente tarefas complexas de previsão de séries temporais, automatizando todo o processo, desde a preparação dos dados até a implantação do modelo. 

O módulo PyCaret Time Series Forecasting oferece suporte a uma ampla gama de métodos de previsão, como ARIMA, Prophet e LSTM. Ele também oferece vários recursos para lidar com valores ausentes, decomposição de séries temporais e visualizações de dados. 
        '''
    )

with col2:
    st.subheader('Imagem Caminhão SemiPesados', divider='red')
    st.image(
        'https://th.bing.com/th/id/OIP.6fvi4BbmYZcIsERb7ApIPAHaE8?rs=1&pid=ImgDetMain',
        use_container_width=True
    )

# Carregando o arquivo Excel local
try:
    data = pd.read_excel(r"C:\Tablets\Semipesados.xlsx")
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
    s.setup(data=data, target='Semipesados', session_id=123)
    st.write("**Configuração inicial do PyCaret concluída.**")

    # Comparar model
    best = s.compare_models()

    # Obter a tabela de comparação
    comparison_df_sp = s.pull()
    st.write("### Comparação de Modelos")
    st.dataframe(comparison_df_sp)

    # Botão para download da tabela de comparação
    csv_sp = comparison_df_sp.to_csv(index=False).encode('utf-8')
    st.download_button(
    "Baixar Comparação",
    data=csv_sp,
    file_name="model_comparison.csv",
    mime="text/csv",
    key="download_button_comparison"  # Chave única
    )

col1, col2, col3  = st.columns([1, 1, 1], gap='large')

with col1:
    st.write('**Time Series - Taget = Caminhões SemiPesados')
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
    st.download_button("Baixar Previsão",
                        data=csv, 
                        file_name="predictions.csv",
                        mime='text/csv',
                        key='download_button_previsão')
    
with col2:
    st.write('Previsão 2025')
    st.metric(label='Previsão 2025', value=35518, delta=324)

with col3:
    st.write('Previsão 2026')
    st.metric(label='Previsão 2026', value=34552, delta=-966)


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
        st.markdown('''### A análise dos resultados fornecidos indica o desempenho de um modelo de árvore de decisão (Decision Tree) combinado com métodos de dessazonalização condicional e detrending. Vamos detalhar os indicadores para avaliar o modelo:

### 1. **Desempenho Geral**
- **MASE (0.3562)** e **RMSSE (0.2615)**: Ambos são métricas escaladas que ajudam a comparar o modelo em relação a um modelo de referência (como uma previsão ingênua). Os valores baixos (geralmente abaixo de 1) indicam que o modelo tem boa capacidade preditiva em relação ao benchmark.
- **Interpretação**: Este modelo supera significativamente uma abordagem ingênua, com erros absolutos médios (MAE) e erros quadráticos médios (RMSE) reduzidos em termos proporcionais.

### 2. **Erros Absolutos e Quadráticos**
- **MAE (197.8341)** e **RMSE (197.8341)**: Aqui, os dois valores são iguais, o que pode indicar que os erros do modelo não são muito dispersos. O RMSE normalmente é maior devido à penalização de erros maiores, mas nesse caso, os erros são uniformes ou os desvios são limitados.
- **Interpretação**: Isso sugere que o modelo é consistente em termos de erro, sem grandes outliers ou previsões muito discrepantes.

### 3. **Erros Percentuais**
- **MAPE (5.96%)** e **SMAPE (5.97%)**: Ambos medem erros relativos em porcentagem, sendo que o SMAPE é simétrico para sobre e subestimações. Com valores próximos de 6%, o modelo demonstra alta precisão, indicando que, em média, as previsões estão a menos de 6% de diferença dos valores reais.
- **Interpretação**: Excelente precisão preditiva, especialmente útil em cenários onde erros relativos são críticos.

### 4. **Tempo de Treinamento**
- **TT (0.036 segundos)**: O tempo de treinamento é extremamente curto, indicando que o modelo é eficiente em termos computacionais.
- **Interpretação**: Ideal para aplicações onde a velocidade de treinamento é uma prioridade.

### **Conclusão**
- Este modelo apresenta bom desempenho em termos de precisão absoluta e relativa, com baixa margem de erro e alta eficiência computacional.
- **Pontos fortes**:
  - Alta precisão (baixos MAPE e SMAPE).
  - Baixo erro escalado (MASE e RMSSE).
  - Tempo de treinamento rápido.
- **Limitações**: O relatório não inclui métricas de validação cruzada ou avaliação em dados fora da amostra. Seria útil confirmar a robustez do modelo em diferentes cenários.

Gostaria de uma análise comparativa com outros modelos ou sugestões para melhorar os resultados?
''')