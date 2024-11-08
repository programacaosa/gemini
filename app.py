import streamlit as st
import google.generativeai as genai
from PIL import Image
import io

# Configuração da API do Google Gemini
genai.configure(api_key="AIzaSyAErnGp3kTtmzP8EpAyLh1vRVq98obquVs")

# Função para gerar conteúdo da imagem com o modelo Gemini
def gerar_resposta(imagem, pergunta):
    try:
        # Carregar o modelo Gemini 1.5 Flash
        model = genai.GenerativeModel('gemini-1.5-flash')  # Alterado para o modelo mais recente
        
        # Gerar conteúdo com a imagem e a pergunta
        response = model.generate_content([pergunta, imagem])  # Passando a pergunta e a imagem
        response.resolve()  # Resolver a resposta

        return response.text  # Retorna a resposta gerada
    except Exception as e:
        return f"Erro ao gerar a resposta: {e}"

# Função principal do Streamlit
def main():
    st.title("API Gemini - Análise de Imagens e Perguntas")
    st.write("Envie uma imagem e faça uma pergunta. A API Gemini irá responder com base no conteúdo da imagem.")

    # Campo para upload da imagem
    uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])
    
    # Campo de pergunta
    pergunta = st.text_input("Pergunte algo sobre a imagem", "Descreva a imagem e quantos animais aparecem nela?")

    if uploaded_file is not None:
        # Exibe a imagem carregada
        imagem_usuario = Image.open(uploaded_file)
        st.image(imagem_usuario, caption="Imagem enviada", use_column_width=True)

        if st.button("Gerar Resposta"):
            # Gerar a resposta com base na imagem e pergunta
            st.write("Gerando resposta... Espere um momento.")
            resposta = gerar_resposta(imagem_usuario, pergunta)
            st.write(f"Resposta: {resposta}")

if __name__ == "__main__":
    main()
