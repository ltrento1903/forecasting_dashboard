
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Estilo para os gráficos
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
dark_style = {
    'figure.facecolor': '#212946',
    'axes.facecolor': '#212946',
    'savefig.facecolor': '#212946',
    'axes.grid': True,
    'axes.grid.which': 'both',
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False,
    'grid.color': '#2A3459',
    'grid.linewidth': '1',
    'text.color': '0.9',
    'axes.labelcolor': '0.9',
    'xtick.color': '0.9',
    'ytick.color': '0.9',
    'font.size': 12,
}
plt.rcParams.update(dark_style)

from pylab import rcParams
rcParams['figure.figsize'] = (18, 7)

# Configuração da página do Streamlit
st.set_page_config(page_title="Análises", layout='wide')

# Layout com 2 colunas para texto e imagem
col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.header("***Análise de Dados***", divider='blue')
    with st.expander('**Explicação da Análise de Dados**'):
        st.markdown('''### Resumo Explicativo das Análises de Dados de Caminhões

Este aplicativo desenvolvido com **Streamlit** foi criado para carregar e analisar dados de caminhões contidos em um arquivo Excel. A análise consiste em explorar as variáveis relacionadas a diferentes tipos de caminhões (como **Pesados**, **Semipesados**, **Médios**, **Leves**, **Semileves** e **Caminhões** gerais) e gerar visualizações interativas para facilitar a interpretação dos dados. A seguir, detalho as etapas e as visualizações presentes no aplicativo:

#### 1. **Carregamento e Visualização Inicial dos Dados**:
   - O arquivo Excel é carregado e os dados são exibidos como uma prévia no Streamlit, permitindo que o usuário veja as primeiras linhas da tabela.
   - O conjunto de dados contém colunas que representam diferentes categorias de caminhões (como mencionado acima), sendo possível visualizar a quantidade ou outras métricas relacionadas a cada tipo.

#### 2. **Derretendo o DataFrame**:
   - Para facilitar a visualização e análise, o DataFrame é transformado (ou "derretido") usando a função melt(). Isso significa que as colunas relacionadas aos tipos de caminhões são convertidas em duas colunas principais: uma para o **tipo de caminhão (Categoria)** e outra para o **valor associado** a esse tipo. 
   - Isso cria um formato mais amigável para análise visual e para a geração de gráficos.

#### 3. **Gráfico de Caixa (Boxplot)**:
   - O gráfico de caixa foi gerado para mostrar a distribuição dos dados em cada categoria de caminhões (Pesados, Semipesados, Médios, Leves, Semileves). Ele é útil para:
     - **Identificar outliers**: Os valores fora do esperado para cada categoria de caminhão.
     - **Visualizar a dispersão**: Como os valores se distribuem (quantidade ou qualquer outra métrica utilizada) para cada tipo de caminhão.
     - **Comparar as categorias**: Como diferentes tipos de caminhões se comparam em termos de valores centrais (mediana) e variação.

   **Como funciona o Boxplot?**
   - O boxplot mostra o **quartil inferior (25%)**, a **mediana (50%)**, o **quartil superior (75%)** e os **outliers** (valores fora da "caixa").
   - Cada tipo de caminhão é representado por uma caixa, permitindo comparações rápidas e visualmente claras entre as categorias.

#### 4. **Histograma**:
   - O histograma foi gerado para exibir a distribuição dos valores de cada categoria de caminhões. O gráfico utiliza a contagem de observações em diferentes faixas de valores para cada categoria.
   - O histograma permite:
     - **Visualizar a distribuição** de valores para cada tipo de caminhão.
     - **Comparar as distribuições** entre os tipos de caminhões em termos de como os dados estão espalhados em várias faixas.
   - Este tipo de gráfico é útil para observar padrões gerais, como quais tipos de caminhões têm valores mais concentrados em determinadas faixas e quais têm uma distribuição mais uniforme.

#### 5. **Interatividade e Visualização com Plotly**:
   - Ambos os gráficos (boxplot e histograma) foram gerados com **Plotly**, uma biblioteca interativa. Isso permite que os usuários explorem os gráficos de maneira dinâmica, podendo:
     - **Passar o mouse sobre os gráficos** para ver os valores exatos.
     - **Interagir com os eixos** para melhorar a leitura de detalhes.
     - **Customizar o gráfico** ao clicar e arrastar ou modificar o zoom.

#### 6. **Objetivo Principal**:
   - O objetivo das análises realizadas é fornecer ao usuário uma visão clara e interativa sobre a distribuição e os padrões nos dados de caminhões.
   - Essas visualizações são essenciais para decisões relacionadas a logística, compras, manutenção e outras áreas que dependem de entender o comportamento e as características dos diferentes tipos de caminhões.

### Conclusão:
Este aplicativo oferece uma análise visual e interativa dos dados relacionados aos caminhões, fornecendo insights valiosos sobre a distribuição e variação dos diferentes tipos de caminhões. Através de gráficos de caixa e histogramas, é possível detectar padrões, identificar outliers e comparar as diferentes categorias de caminhões, facilitando a tomada de decisões e a análise estratégica.''')

with col2:
    st.header('Imagem Caminhões', divider='blue')
    st.image('https://th.bing.com/th/id/OIP.8m82JUxc33LPL2VX97KgGAHaEo?rs=1&pid=ImgDetMain', use_container_width=True)

# Carregar os dados
file_path = r"C:\Tablets\Caminhão_Total_famílias.xlsx"
try:
    df = pd.read_excel(file_path)

    # Layout com 3 colunas
    col1, col2, col3 = st.columns([1, 1, 1], gap='large')

    with col1:
        st.success("Base de Dados Carregada com Sucesso")
        st.write("Prévia dos Dados:")
        st.write(df)

        # Derreter o DataFrame para que as colunas desejadas apareçam em uma coluna única
        df_melted = df[['Caminhões', 'Pesados', 'Semipesados', 'Médios', 'Leves', 'Semileves']].melt(var_name='Categoria', value_name='Valor')

    with col2:
        st.write("**Gráfico de Caixa**")
        fig = px.box(df_melted, x='Categoria', y='Valor', color='Categoria', 
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig)

    with col3:
        # Ordenar e calcular porcentagem acumulada para gráfico de Pareto
        df_melted_sorted = df_melted.groupby('Categoria', as_index=False).sum()
        df_melted_sorted = df_melted_sorted.sort_values(by='Valor', ascending=False)
        df_melted_sorted['Porcentagem Acumulada'] = df_melted_sorted['Valor'].cumsum() / df_melted_sorted['Valor'].sum() * 100

        # Criar o gráfico de Pareto
        fig_h = px.bar(df_melted_sorted, x='Categoria', y='Valor', 
                       color='Categoria', color_discrete_sequence=px.colors.qualitative.Set2, text='Valor')
        fig_h.add_scatter(x=df_melted_sorted['Categoria'], 
                          y=df_melted_sorted['Porcentagem Acumulada'], 
                          mode='lines+markers', 
                          name='Porcentagem Acumulada',
                          yaxis='y2')

        fig_h.update_layout(
            title='Gráfico de Pareto',
            yaxis=dict(title='Valor'),
            yaxis2=dict(title='Porcentagem Acumulada', overlaying='y', side='right', showgrid=False),
            legend=dict(title='Legenda'),
            xaxis=dict(title='Categoria')
        )

        st.write("**Gráfico de Pareto**")
        st.plotly_chart(fig_h)

except FileNotFoundError:
    st.error(f"Arquivo não encontrado: {file_path}")
except Exception as e:
    st.error(f"Erro ao carregar ou processar os dados: {e}")



