from langchain_community.document_loaders import TextLoader as LangLoader
class TextLoader:
  def __init__(self):
    documents = LangLoader("./data/dream.txt").load()
    print(documents)
  

if __name__ == "__main__":
    loader = TextLoader()