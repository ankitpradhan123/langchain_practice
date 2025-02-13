from langchain_ollama.chat_models import ChatOllama
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] =  os.getenv("GROQ_API_KEY")

class ChatLlmFactory:
    def __init__(self, model_name: str, library: str) -> None:
        self.model_name = model_name
        self.library = library

    def generate_llm(self) -> object:
        library = self.library
        if library == "ChatGroq":
            llm_obj = ChatGroq(model=self.model_name)
        elif library == "ChatOllama":
            llm_obj = ChatOllama(model=self.model_name)
        else:
            llm_obj = ChatGroq(model="deepseek-r1-distill-llama-70b")
        return llm_obj



