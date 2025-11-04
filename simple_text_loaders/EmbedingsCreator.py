from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import  os

class EmbedingsCreator:
    def __init__(self):
       load_dotenv()
       self.Ollama_Url=os.getenv("ollama_url")
       self.embedding_model = os.getenv("embedding_model")

    def create_embedings(self,text):
        embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.Ollama_Url
        )
        vector = embeddings.embed_query(text)
        return embeddings