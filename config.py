import os
import uuid

from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

cwd = os.getcwd()

'''.env config
'''
load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY = "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"

'''Constant config
'''
ERROR_MESSAGE = 'We are facing technical issue at this moment.'

'''Qdrant config
'''
# QDRANT_URL = 'http://localhost:6333'
QDRANT_URL = 'https://a5993180-1deb-4616-880d-d61e5caeb676.europe-west3-0.gcp.cloud.qdrant.io'
JWT_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhY2Nlc3MiOiJtIn0.eVcMqxeuPvG2O9aY1275XEN4ordz26AlK_e-eB2YFdc"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.SPPxvQrCR6meVMnFd_zh2kqRXpTLRtPuMmwvOpnQWSY"
client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    timeout=30,
)
COLLECTION_NAME = f'{uuid.uuid1()}'

'''Embeddings config
'''
# print(torch.cuda.is_available())
EMBEDDING_FUNCTION = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2',
    model_kwargs={'device': 'cpu'}
)

'''Dir config
'''
DOWNLOAD_DIR = 'arxiv_pdfs'
DOWNLOAD_DIR_PATH = os.path.join(
    cwd,
    DOWNLOAD_DIR
)
os.makedirs(DOWNLOAD_DIR_PATH, exist_ok=True)
