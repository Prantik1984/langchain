from langchain_community.document_loaders import TextLoader as LangLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from EmbedingsCreator import EmbedingsCreator
import re
import  os
class TextLoader:
  def __init__(self):
    load_dotenv()
    documents = LangLoader("./data/dream.txt").load()
    cleaned_documents=[self.clean_text(doc.page_content) for doc in documents]
    chunk_size=int(os.getenv("CHUNK_SIZE"))
    chunk_overlap=int(os.getenv("chunk_overlap"))
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    texts=[self.clean_text(text.page_content) for text in texts]

    embedder = EmbedingsCreator(texts)
    embedings = embedder.create_embedings()
    print(embedings)

  def clean_text(self,txt):
    txt = re.sub(r"[^a-zA-Z\s]", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    txt=txt.lower()
    return txt

if __name__ == "__main__":
    loader = TextLoader()