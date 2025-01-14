# Importando bibliotecas necessárias
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
import streamlit as st
#CAMINHÕES TOTAL
# Configuração da página do Streamlit
st.set_page_config(
    page_title='Forecast Licenciamentos Caminhões Leves',
    layout='wide',
    initial_sidebar_state='auto'
)

#CAMINHÕES LEVES

st.header("Forecast Licenciamentos de Caminhões Leves", divider='grey')

# Layout principal
col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.subheader('Forecast Licenciamentos Caminhões Leves por meio do PyCaret***', divider='grey')
    st.markdown('''O segmento de licenciamentos de caminhões leves, monitorado pela Anfavea, inclui veículos com capacidade de carga entre 3,5 e 6 toneladas. Esses caminhões são essenciais para o transporte de mercadorias em áreas urbanas e rurais, oferecendo flexibilidade e eficiência logística. Eles são amplamente utilizados por empresas de logística, comércio e construção, devido à sua capacidade de transportar cargas moderadas com agilidade e economia.
    
O módulo PyCaret Time Series é uma ferramenta avançada para analisar e prever dados de séries temporais usando aprendizado de máquina e técnicas estatísticas clássicas. Esse módulo permite que os usuários executem facilmente tarefas complexas de previsão de séries temporais, automatizando todo o processo, desde a preparação dos dados até a implantação do modelo. 

O módulo PyCaret Time Series Forecasting oferece suporte a uma ampla gama de métodos de previsão, como ARIMA, Prophet e LSTM. Ele também oferece vários recursos para lidar com valores ausentes, decomposição de séries temporais e visualizações de dados. 
        '''
    )

with col2:
    st.subheader('Imagem Caminhão Leves', divider='grey')
    st.image(
        'https://th.bing.com/th/id/OIP.aHxCC_IE3d66uFRCHGgKNgHaDu?rs=1&pid=ImgDetMain',
        use_container_width=True
    )

# Carregando o arquivo Excel local
try:
    data = pd.read_excel(r"C:\Tablets\Leves.xlsx")
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
    s.setup(data=data, target='Leves', session_id=123)
    st.write("**Configuração inicial do PyCaret concluída.**")

    # Comparar model
    best = s.compare_models()

    # Obter a tabela de comparação
    comparison_df_le = s.pull()
    st.write("### Comparação de Modelos")
    st.dataframe(comparison_df_le)

    # Botão para download da tabela de comparação
    csv_me = comparison_df_le.to_csv(index=False).encode('utf-8')
    st.download_button(
    "Baixar Comparação",
    data=csv_me,
    file_name="model_comparison.csv",
    mime="text/csv",
    key="download_button_comparison_le"  # Chave única
    )

col1, col2, col3  = st.columns([1, 1, 1], gap='large')

with col1:
    st.write('**Time Series - Taget = Caminhões Leves')
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

col1, col2, col3=st.columns([1,1,2], gap='large')

with col1:
    st.write('**Previsâo da Amostra**')
    s.plot_model(final_best, plot='insample', display_format='streamlit')

with col2:
    st.write('**Diagnósticos**')
    s.plot_model(final_best, plot='residuals', display_format='streamlit')

with col3:
    st.write('**Decomposição')
    s.plot_model(final_best, plot='decomp_stl', display_format='streamlit')

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
                        key='download_button_previsão_le')
    
with col2:
    st.write('Previsão 2025')
    st.metric(label='Previsão 2025', value=9557, delta=-340)

with col3:
    st.write('Previsão 2026')
    st.metric(label='Previsão 2026', value=9124, delta=-433)    

col1, col2=st.columns([1,1], gap='large')

with col1:
    with st.expander('Exponential Smoothing'):
        st.markdown('''### A análise do modelo **Exponential Smoothing** com configuração **(seasonal='mul', sp=12, trend='add')** está detalhada a seguir:
---
### **Modelo de Suavização Exponencial**
O modelo **Exponential Smoothing (ETS)** é uma abordagem robusta para séries temporais, combinando três componentes principais: **Erro**, **Tendência** e **Sazonalidade**. Sua configuração com parâmetros específicos é interpretada assim:

1. **Seasonal='mul'**  
   - Define a sazonalidade como **multiplicativa**, onde o impacto da sazonalidade varia proporcionalmente à magnitude da série.  
   - **Exemplo**: Em séries de maior escala, a sazonalidade será mais pronunciada.  

2. **sp=12**  
   - Período sazonal é de 12, indicando padrões repetitivos anuais com dados mensais.  
   - **Relevância**: Adequado para séries temporais com periodicidade anual.

3. **Trend='add'**  
   - Define a tendência como **aditiva**, ou seja, o crescimento ou decrescimento ocorre a uma taxa constante ao longo do tempo.  
   - **Impacto**: Modela séries com crescimento linear, sem aceleração ou desaceleração exponencial.

---
2. **Ajuste aos Dados**:  
   - A sazonalidade multiplicativa é ideal para séries onde a magnitude da variação sazonal aumenta ou diminui proporcionalmente aos valores médios da série.
   - A tendência aditiva permite capturar mudanças constantes no nível da série.

3. **Capacidade de Previsão**:  
   - Este modelo é adequado para séries com padrões sazonais bem definidos e tendência linear.  
   - Limitações podem surgir em séries com sazonalidade irregular ou tendências não lineares.

---

### **Vantagens do Modelo**

1. **Interpretação Simples**:  
   - Fácil de entender e implementar, com parâmetros intuitivos.
2. **Flexibilidade**:  
   - Suporta diferentes combinações de tendência e sazonalidade (aditiva/multiplicativa).  
3. **Adequação a Dados Sazonais**:  
   - A configuração (seasonal='mul', sp=12) é apropriada para dados com periodicidade anual bem definida.  

---

### **Desvantagens do Modelo**

1. **Assume Linearidade da Tendência**:  
   - A tendência aditiva não é adequada para séries com crescimento ou decrescimento exponencial.  

2. **Sazonalidade Fixa**:  
   - O modelo assume que a sazonalidade é constante ao longo do tempo, o que pode não ser verdadeiro para séries onde padrões sazonais mudam dinamicamente.

3. **Sensibilidade a Dados Ruidosos**:  
   - Pode ser afetado por valores atípicos ou séries com alta variabilidade.  

---

### **Quando Usar este Modelo?**

- **Séries Temporais com Tendência Linear**: Como vendas mensais que crescem de forma constante.  
- **Séries com Sazonalidade Proporcional**: Onde os efeitos sazonais variam em magnitude com a série.  
- **Curto a Médio Prazo**: O modelo é mais confiável para previsões de curto prazo.  

---

### **Conclusão**
O modelo **Exponential Smoothing (seasonal='mul', sp=12, trend='add')** é uma escolha robusta e eficiente para séries temporais com tendência linear e sazonalidade proporcional à magnitude. No entanto, para séries com características mais complexas (ex.: sazonalidade variável ou tendências não lineares), pode ser necessário explorar alternativas como **ARIMA** ou **Prophet**.
''', unsafe_allow_html=True)
   
with col2:
    with st.expander('Análise dos Resultados Exponential Smoothing'):
        st.markdown('''A análise do desempenho do modelo **Exponential Smoothing**, com base nas métricas apresentadas, é detalhada abaixo:

---

### **Métricas Avaliadas**

1. **MASE (Mean Absolute Scaled Error)**  
   - **Valor reportado**: 0.1267  
   - **Interpretação**: O MASE compara o desempenho do modelo com um modelo de referência simples, como o Naïve.  
     - Valor < 1: O modelo supera o Naïve.
     - Valor > 1: O modelo é inferior ao Naïve.  
   - **Análise**: Um MASE de 0.1267 é excepcionalmente bom, indicando que o modelo supera amplamente o baseline.

---

2. **RMSSE (Root Mean Squared Scaled Error)**  
   - **Valor reportado**: 0.0899  
   - **Interpretação**: Métrica semelhante ao MASE, mas penaliza erros maiores com mais intensidade.  
   - **Análise**: Um RMSSE tão baixo reforça a eficácia do modelo ao minimizar grandes desvios.

---

3. **MAE (Mean Absolute Error)**  
   - **Valor reportado**: 40.5113  
   - **Interpretação**: O erro absoluto médio reflete o desvio médio entre os valores previstos e os reais, em termos absolutos.  
   - **Análise**: O valor absoluto parece aceitável, mas sua significância depende da escala dos dados.

---

4. **RMSE (Root Mean Squared Error)**  
   - **Valor reportado**: 40.5113  
   - **Interpretação**: Penaliza desvios maiores mais severamente que o MAE.  
   - **Análise**: O RMSE igual ao MAE sugere que os desvios são consistentes, sem grandes outliers.

---

5. **MAPE (Mean Absolute Percentage Error)**  
   - **Valor reportado**: 0.0435 (4.35%)  
   - **Interpretação**: Representa o erro médio absoluto como uma porcentagem dos valores reais.  
   - **Análise**: Um MAPE de 4.35% é excelente e indica previsões bastante precisas em relação à magnitude da série.

---

6. **SMAPE (Symmetric Mean Absolute Percentage Error)**  
   - **Valor reportado**: 0.0447 (4.47%)  
   - **Interpretação**: Métrica simétrica que evita distorções causadas por valores extremos.  
   - **Análise**: O SMAPE confirma a precisão do modelo com desvios mínimos.

---

7. **TT (Tempo Total)**  
   - **Valor reportado**: 0.0667 segundos  
   - **Interpretação**: Tempo computacional necessário para treinar e prever usando o modelo.  
   - **Análise**: O modelo é altamente eficiente, com tempo de execução extremamente baixo.

---

### **Pontos Fortes**
1. **Desempenho Excepcional**:
   - O MASE e RMSSE extremamente baixos indicam que o modelo é muito eficaz em prever a série.
   - O MAPE e SMAPE abaixo de 5% mostram precisão excelente, mesmo em termos relativos.

2. **Consistência nos Erros**:  
   - A igualdade entre MAE e RMSE sugere que os erros são consistentes, sem grandes desvios ou outliers.

3. **Alta Eficiência Computacional**:
   - O tempo total de execução (TT = 0.0667 segundos) é praticamente instantâneo, adequado para aplicações em tempo real ou de grande escala.

---

### **Pontos Fracos**
1. **Interpretação Escalar**:
   - O MAE e RMSE (40.5113) podem ser considerados altos dependendo da escala dos dados, mas isso não é necessariamente uma limitação do modelo.

2. **Sazonalidade e Tendência**:
   - O desempenho depende de como o modelo foi configurado para capturar a sazonalidade e tendência (não especificado aqui).  
   - Séries com padrões complexos podem exigir ajustes adicionais.

---

### **Conclusão**
O modelo **Exponential Smoothing** apresentou resultados notavelmente bons, destacando-se como uma solução eficaz e computacionalmente eficiente para séries temporais simples e bem definidas. Com métricas como MAPE (4.35%) e SMAPE (4.47%), é uma opção confiável para previsões em aplicações práticas.

#### **Recomendações**
- **Escala dos Dados**: Verificar se o MAE e RMSE são aceitáveis na escala da série.
- **Validação Adicional**: Confirmar o desempenho em diferentes janelas temporais ou conjuntos de dados.
- **Exploração de Alternativas**: Em caso de padrões mais complexos, considerar modelos complementares como ARIMA, Prophet ou LSTMs.
''')