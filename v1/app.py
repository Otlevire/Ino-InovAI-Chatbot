import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import utils


HUGGINGFACEHUB_API_TOKEN = utils.HUGGINGFACEHUB_API_TOKEN
PINECONE_API_KEY = utils.PINECONE_API_KEY

# Load Documents
def load_docs(directory="Docs/"):
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    return documents

# Transform(Split) Documents
def split_docs(documents, chunk_size=2000, chunk_overlap=250):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n", "\n\n","\n\n\n"], chunk_size=chunk_size, chunk_overlap=chunk_overlap) # Break large documents in few chunks
    docs = text_splitter.split_documents(documents=documents)
    return docs


def get_similar_docs(index, query, k=2):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs


def get_answer(relevant_docs, query, chain):
    response = chain.run(input_documents=relevant_docs, question=query)
    return response


# Load
directory = 'Docs/'
documents = load_docs(directory)
print(f"Num docs(pages):{len(documents)}")
# print(documents)

#Transform
docs = split_docs(documents)
print(f"Num docs separados:{len(docs)}")

#Embedding
# embeddings = OpenAIEmbeddings(model_name="ada")
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")

query_result = embeddings.embed_query("Hello Buddy")
# print(len(query_result))

pinecone.init(api_key=utils.PINECONE_API_KEY,environment=utils.PINECONE_ENVIRONMENT)
index = Pinecone.from_documents(docs, embeddings, index_name=utils.PINECONE_INDEX_NAME)

# llm = HuggingFaceHub(repo_id="bigscience/bloom",model_kwargs={"temperature":1e-10})
llm = HuggingFaceHub(repo_id="google/flan-t5-large",model_kwargs={"temperature":0.9})
chain = load_qa_chain(llm, chain_type="stuff")

try:
    while True:
        our_query = input("O que deseja saber:")
        if our_query.lower() == 'exit':
            raise KeyboardInterrupt("saindo")
        
        relevant_docs = get_similar_docs(index=index, query=our_query, k=2)
        print(f"Documentos Base da resposta:{relevant_docs}")
        answer = get_answer(chain=chain,query=our_query, relevant_docs=relevant_docs)
        print(f"Resposta:{answer}")
except KeyboardInterrupt:
    print("\n\n")
    print("Saindo e finalizando o programa!")
