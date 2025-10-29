from langchain_community.document_loaders import TextLoader as LangLoader
import re
class TextLoader:
  def __init__(self):
    documents = LangLoader("./data/dream.txt").load()
    cleaned_documents=[self.clean_text(doc.page_content) for doc in documents]
    print(cleaned_documents)
  
  def clean_text(self,txt):
    txt = re.sub(r"[^a-zA-Z\s]", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    txt=txt.lower()
    return txt

if __name__ == "__main__":
    loader = TextLoader()