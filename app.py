import streamlit as st

# Configurações gerais da página
st.set_page_config(
    page_title="Caminhões - Produtos: Pesados, Semipesados, Médios, Leves e Semileves",
    page_icon="🚚",
    layout="wide"
)

# Conteúdo principal da página inicial
st.title("Bem-vindo ao Dashboard de Licenciamentos Caminhões! 🚚")


st.markdown("""
Este dashboard contém:
- **Licenciamentos históricos**: Uma visão detalhada dos licenciamentos desde 2012.
- **Análises detalhadas**: Análise Exploratória.
- **Previsões**: Projeções futuras por meio de modelos estatísticos avançados.

**Dica:** Use o menu lateral para navegar entre as páginas.
""")

# Adicione uma imagem decorativa
st.image(
    "https://4.bp.blogspot.com/-Q-EPuOveyNo/TmjA6N7edFI/AAAAAAAAABM/W8SO7iA158k/s1600/Pack_Shot_Estrela.jpg",
    caption="Caminhões",
    use_container_width=True
)
