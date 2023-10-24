import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import os

load_dotenv()
Docs="Docs/"
OPENAI_API_KEY=str(os.getenv('OPENAI_API_KEY'))
chat_history = []

def get_pdf_text(directory):
    text = ""
    # Listar todos os arquivos no diretório
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):  # Verificar se o arquivo é um PDF
                pdf_reader = PdfReader(os.path.join(directory, filename))
                for page in pdf_reader.pages:
                    text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks



def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectorstore.as_retriever(),memory=memory)
    return conversation_chain


def get_answer(chat, user_question):
    global chat_history
    response = chat({'question': user_question})
    # response = final_chat({'question': user_question})
    chat_history = response['chat_history']

    return response

def main():
    # get pdf text
    raw_text = get_pdf_text(Docs)

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)
    print(len(text_chunks))

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # create conversation chain
    final_chat = get_conversation_chain(vectorstore)
    # print(type(final_chat))
    try:
        while(True):
            question = input("\n\nO que deseja saber:")
            if question.lower() == 'exit':
                raise KeyboardInterrupt("saindo")
            
            answer = get_answer(chat=final_chat,user_question=question)
            print(f"Resposta:{answer['answer']}")
        
    except KeyboardInterrupt:
        print("\n\nEncerrando o Programa!")
if __name__ == '__main__':
    main()