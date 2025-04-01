from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
# from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# import feedparser
# import faiss
import json
import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain.vectorstores.qdrant import Qdrant
import config

# chat_model = ChatOpenAI(
#     temperature=0,
#     openai_api_key="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",
#     model_name='gpt-4o'
# )

class ArxivModel:
    def __init__(self):

        self.store = {}
        # TODO: make this dynamic for new sessions via the app
        self.session_config = {"configurable": {"session_id": "0"}}

    def _set_api_keys(self):
        # load all env vars from .env file
        load_dotenv()

        # Add all such vars in OS env vars
        for key, value in os.environ.items():
            if key in os.getenv(key):  # Check if it exists in the .env file
                os.environ[key] = value

        print("All environment variables loaded successfully!")

    def load_json(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def create_documents(self, data):
        docs = []
        for paper in data:
            title = paper["title"]
            abstract = paper["summary"]
            link = paper["link"]
            paper_content = f"Title: {title}\nAbstract: {abstract}"
            paper_content = paper_content.lower()

            docs.append(Document(page_content=paper_content,
                                 metadata={"link": link}))

        return docs

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def create_retriever(self):
        # Load a pre-trained embedding model
        # embedding_model = CohereEmbeddings(model="embed-english-light-v3.0")

        # index = faiss.IndexFlatL2(
        #     len(embedding_model.embed_query("Hello LLM")))

        # vector_db = FAISS(
        #     embedding_function=embedding_model,
        #     index=index,
        #     docstore=InMemoryDocstore(),
        #     index_to_docstore_id={},
        # )

        # vector_db.add_documents(docs)

        # self.retriever = vector_db.as_retriever()
        vector_db = Qdrant(client=config.client, embeddings=config.EMBEDDING_FUNCTION,
                           collection_name=st.session_state.COLLECTION_NAME)

        self.retriever = vector_db.as_retriever()

    def get_history_aware_retreiver(self):
        system_prompt_to_reformulate_input = (
            "Given a chat history and the latest user question "
            "which might reference the context in the chat history "
            "formulate a standalone question which can be understood "
            "without chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        prompt_to_reformulate_input = ChatPromptTemplate.from_messages([
            ("system", system_prompt_to_reformulate_input),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever_chain = create_history_aware_retriever(
            self.llm, self.retriever, prompt_to_reformulate_input
        )
        return history_aware_retriever_chain

    def get_prompt(self):
        # system_prompt = ("You are an AI assistant called 'ArXiv Assist' that has a conversation with the user and answers to questions. "
        #                  "Try looking into the research papers content provided to you to respond back. If you could not find any relevant information there, mention something like 'I do not have enough information form the research papers. However, this is what I know...' and then try to formulate a response by your own. "
        #                  "There could be cases when user does not ask a question, but it is just a statement. Just reply back normally and accordingly to have a good conversation (e.g. 'You're welcome' if the input is 'Thanks'). "
        #                  "If you mention the name of a paper, provide an arxiv link to it. "
        #                  "Be polite, friendly, and format your response well (e.g., use bullet points, bold text, etc.). "
        #                  "Below are relevant excerpts from the research papers:\n{context}\n\n"
        #                  )
        system_prompt= ("You are an AI assistant named 'ArXiv Assist' that helps users understand and explore a single academic research paper. "
                        "You will be provided with content from one research paper only. Treat this paper as your only knowledge source. "
                        "Your responses must be strictly based on this paper's content. Do not use general knowledge or external facts unless explicitly asked to do so — and clearly indicate when that happens. "
                        "If the paper does not provide enough information to answer the user’s question, respond with: 'I do not have enough information from the research paper. However, this is what I know…' and then answer carefully based on your general reasoning. "
                        "Avoid speculation or assumptions. Be precise and base your answers on what the paper actually says. "
                        "When possible, refer directly to phrases or ideas from the paper to support your explanation. "
                        "If summarizing a section or idea, use clean formatting such as bullet points, bold terms, or brief section headers to improve readability. "
                        "There could be cases when user does not ask a question, but it is just a statement. Just reply back normally and accordingly to have a good conversation (e.g. 'You're welcome' if the input is 'Thanks'). "
                        "Always be friendly, helpful, and professional in tone."
                        "\n\nHere is the content of the paper you are working with:\n{context}\n\n")

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "Answer the following question: {input}")
        ])

        return prompt

    def create_conversational_rag_chain(self):
        # Subchain 1: Create ``history aware´´ retriever chain that uses conversation history to update docs
        history_aware_retriever_chain = self.get_history_aware_retreiver()

        # Subchain 2: Create chain to send docs to LLM
        # Generate main prompt that takes history aware retriever
        prompt = self.get_prompt()
        # Create the chain
        qa_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)

        # RAG chain: Create a chain that connects the two subchains
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever_chain,
            combine_docs_chain=qa_chain)

        # Conversational RAG Chain: A wrapper chain to store chat history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        return conversational_rag_chain

    def get_model(self):
        # docs = self.create_documents(data)
        self.create_retriever()
        # self.llm = ChatCohere(
        #     model="command-r-plus-08-2024", max_tokens=256, temperature=0.5)
        self.llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-002")
        # self.llm= chat_model
        conversational_rag_chain = self.create_conversational_rag_chain()
        return conversational_rag_chain, self.session_config

