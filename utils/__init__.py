import os
from dotenv import load_dotenv
load_dotenv()



# OPENAI_API_KEY = str(os.getenv('OPENAI_API_KEY'))
HUGGINGFACEHUB_API_TOKEN = str(os.getenv('HUGGINGFACEHUB_API_TOKEN'))
PINECONE_API_KEY = str(os.getenv('PINECONE_API_KEY'))

PINECONE_INDEX_NAME="inovai-site"
PINECONE_ENVIRONMENT ="gcp-starter"
