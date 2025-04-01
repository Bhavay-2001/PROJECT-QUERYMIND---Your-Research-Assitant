from langchain.schema import Document
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.vectorstores.qdrant import Qdrant

import os
import json
import config
from dotenv import load_dotenv

class ArxivModel:
    store = {}  # shared across instances

    def __init__(self, session_id: str, collection_name: str):
        self.session_id = session_id
        self.collection_name = collection_name
        self.session_config = {"configurable": {"session_id": self.session_id}}

    def _set_api_keys(self):
        load_dotenv()
        for key, value in os.environ.items():
            if key in os.getenv(key):
                os.environ[key] = value

    def load_json(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def create_documents(self, data):
        docs = []
        for paper in data:
            title = paper["title"]
            abstract = paper["summary"]
            link = paper["link"]
            paper_content = f"Title: {title}\nAbstract: {abstract}".lower()
            docs.append(Document(page_content=paper_content, metadata={"link": link}))
        return docs

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in ArxivModel.store:
            ArxivModel.store[session_id] = ChatMessageHistory()
        return ArxivModel.store[session_id]

    def create_retriever(self):
        vector_db = Qdrant(
            client=config.client,
            embeddings=config.EMBEDDING_FUNCTION,
            collection_name=self.collection_name
        )
        self.retriever = vector_db.as_retriever()

    def get_history_aware_retreiver(self):
        reformulation_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Given a chat history and a follow-up question, rephrase it into a standalone question. "
             "If already standalone, return as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        return create_history_aware_retriever(
            self.llm,
            self.retriever,
            reformulation_prompt
        )

    def get_prompt(self):
        system_prompt = (
            "You are an AI assistant named 'ArXiv Assist' that helps users understand and explore a single academic research paper. "
            "You will be provided with content from one research paper only. Treat this paper as your only knowledge source. "
            "Your responses must be strictly based on this paper's content. Do not use general knowledge or external facts unless explicitly asked to do so — and clearly indicate when that happens. "
            "If the paper does not provide enough information to answer the user’s question, respond with: 'I do not have enough information from the research paper. However, this is what I know…' and then answer carefully based on your general reasoning. "
            "Avoid speculation or assumptions. Be precise and base your answers on what the paper actually says. "
            "When possible, refer directly to phrases or ideas from the paper to support your explanation. "
            "If summarizing a section or idea, use clean formatting such as bullet points, bold terms, or brief section headers to improve readability. "
            "There could be cases when user does not ask a question, but it is just a statement. Just reply back normally and accordingly to have a good conversation (e.g. 'You're welcome' if the input is 'Thanks'). "
            "Always be friendly, helpful, and professional in tone."
            "\n\nHere is the content of the paper you are working with:\n{context}\n\n"
        )

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "Answer the following question: {input}")
        ])

    def create_conversational_rag_chain(self):
        retriever_chain = self.get_history_aware_retreiver()
        prompt = self.get_prompt()

        qa_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
            document_variable_name="context"
        )

        rag_chain = create_retrieval_chain(
            retriever=retriever_chain,
            combine_docs_chain=qa_chain
        )

        return RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    def get_model(self):
        self.create_retriever()
        self.llm= ChatGroq(groq_api_key=config.GROQ_API_KEY, model_name="Gemma2-9b-it")
        rag_chain = self.create_conversational_rag_chain()
        return rag_chain, self.session_config
