from langchain_core.documents import Document
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_core.prompts import ChatPromptTemplate
from llm import ChatLlmFactory
from typing import Union, List, Tuple
import os
import pickle
import constants
import datetime

# Ensure directories exist
os.makedirs(constants.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(constants.FAISS_INDEX_PATH, exist_ok=True)

class Rag:
    def __init__(self, pdf_path: str, pdf_name: str) -> None:
        self.pdf_path = pdf_path
        self.pdf_name = pdf_name

    def process_pdf(self) -> VectorStore:
        print(f"Processing pdfs started {datetime.datetime.now()}")
        embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
        docs = PyPDFLoader(self.pdf_path).load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        chunks = text_splitter.split_documents(docs)
        chunks = chunks[:5] #Can remove this line
        db = FAISS.from_documents(chunks, embeddings)
        self.save_vector(db)
        print(f"Processing pdfs completed {datetime.datetime.now()}")
        return db

    @classmethod
    def retrieve_context(cls, vector_store: VectorStore, query: str) -> Tuple[str, List[Document]]:
        documents = vector_store.similarity_search(query)
        context = "\n\n".join(document.page_content for document in documents)
        return context, documents

    def save_vector(self, vector_store: VectorStore) -> None:
        vector_store.save_local(constants.FAISS_INDEX_PATH)

    @classmethod
    def generate_response(cls, query: str, context: str) -> str:
        prompt_template = """
                    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the anser concise.
                    Question: {query}
                    Context: {context}
                    Answer:
                    """

        llm_factory = ChatLlmFactory("deepseek-r1-distill-llama-70b", "ChatGroq")
        llm = llm_factory.generate_llm()
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        result = chain.invoke({"query": query, "context": context})
        return result.content

    @classmethod
    def load_vector(cls) -> Union[VectorStore, None]:
        if os.path.exists(constants.FAISS_INDEX_FILE):
            embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
            vector_store = FAISS.load_local(constants.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            return vector_store
        return None








