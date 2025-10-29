from langchain_community.document_loaders import TextLoader as LangLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import  os
class TextLoader:
  def __init__(self):
    load_dotenv()
    documents = LangLoader("./data/dream.txt").load()
    cleaned_documents=[self.clean_text(doc.page_content) for doc in documents]
    chunk_size=int(os.getenv("CHUNK_SIZE"))
    chunk_overlap=int(os.getenv("chunk_overlap"))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
    # texts = text_splitter.split_documents(documents)
    # print(texts)

  def clean_text(self,txt):
    txt = re.sub(r"[^a-zA-Z\s]", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    txt=txt.lower()
    return txt

if __name__ == "__main__":
    loader = TextLoader()