import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

# Carregar o modelo Universal Sentence Encoder
@st.cache(allow_output_mutation=True)
def load_model():
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    return model

model = load_model()

# Carregar os arquivos .txt da pasta 'treino'
@st.cache
def load_knowledge_base_from_files(directory):
    knowledge_base = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                knowledge_base[filename] = content  # Usando o nome do arquivo como chave
    return knowledge_base

# Caminho da pasta onde os arquivos de treino estão localizados
directory = "treino"  # Altere para o caminho correto se necessário
knowledge_base = load_knowledge_base_from_files(directory)
knowledge_texts = list(knowledge_base.values())

# Calcular os embeddings para a base de conhecimento
knowledge_embeddings = model(knowledge_texts)

# Função para encontrar a resposta mais próxima
def get_response(question):
    question_embedding = model([question])
    
    # Convertendo os embeddings para arrays numpy
    question_embedding = question_embedding.numpy()
    knowledge_embeddings_numpy = knowledge_embeddings.numpy()
    
    # Normalizando os embeddings para evitar que a magnitude dos vetores interfira no cálculo de similaridade
    question_embedding_norm = question_embedding / np.linalg.norm(question_embedding)
    knowledge_embeddings_norm = knowledge_embeddings_numpy / np.linalg.norm(knowledge_embeddings_numpy, axis=1)[:, np.newaxis]
    
    # Cálculo da similaridade
    similarity_scores = np.dot(knowledge_embeddings_norm, question_embedding_norm.T)  # Cálculo da similaridade por produto escalar
    best_match_index = np.argmax(similarity_scores)
    best_match_score = similarity_scores[best_match_index]
    
    # Se a similaridade for alta, retorna a resposta, caso contrário, retorna uma mensagem de erro
    if best_match_score > 0.5:  # Limite para considerar uma boa correspondência
        response = knowledge_base[list(knowledge_base.keys())[best_match_index]]
    else:
        response = "Desculpe, não sei a resposta."
    
    return response

# Interface do Streamlit
st.title("Chatbot que Aprende com TXT 📚")
st.write("Faça perguntas com base nos textos carregados:")

question = st.text_input("Digite sua pergunta:")

if question:
    response = get_response(question)
    st.write(f"**Resposta:** {response}")
else:
    st.write("Aguardando perguntas...")
