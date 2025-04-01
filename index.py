from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_qdrant import QdrantVectorStore

import os
import streamlit as st
import config

def create_index(dir_path: str) -> str:
    st.write('📦 Started vector generation process.')

    loader = DirectoryLoader(
        dir_path,
        glob='**/*.pdf',
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3072,
        chunk_overlap=64
    )
    texts = text_splitter.split_documents(documents)

    qdrant_client = QdrantClient(
        url=config.QDRANT_URL,
        api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.SPPxvQrCR6meVMnFd_zh2kqRXpTLRtPuMmwvOpnQWSY',  # moved to config for best practice
        prefer_grpc=False,
        timeout=30
    )

    existing = qdrant_client.get_collections().collections
    if st.session_state.COLLECTION_NAME not in [c.name for c in existing]:
        qdrant_client.create_collection(
            collection_name=st.session_state.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE
            )
        )

    Qdrant.from_documents(
        documents=texts,
        embedding=config.EMBEDDING_FUNCTION,
        collection_name=st.session_state.COLLECTION_NAME,
        url=config.QDRANT_URL,
        api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.SPPxvQrCR6meVMnFd_zh2kqRXpTLRtPuMmwvOpnQWSY',
        prefer_grpc=False,
        force_recreate=True,
    )

    return "✅ Documents uploaded and index created successfully. You can chat now."

 