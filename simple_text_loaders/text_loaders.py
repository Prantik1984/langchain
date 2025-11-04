from langchain_community.document_loaders import TextLoader as LangLoader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from EmbedingsCreator import EmbedingsCreator
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

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
    embedder = EmbedingsCreator()
    for text in texts:
      embedings = embedder.create_embedings(text)

    retriever = FAISS.from_texts(texts,embedings).as_retriever(search_kwargs={"k": 2})

    query = "Give me a summary of the speech in bullet points"
    docs = retriever.invoke(query)

    prompt = ChatPromptTemplate.from_template(
       "Please use the following docs:\n{docs}\n\n"
       "to answer the question:\n{query}"
    )

    llm = ChatOllama(model="llama3.2:latest")
    chain = prompt | llm| StrOutputParser()
    response = chain.invoke({"docs": docs, "query": query})
    print(f"Model Response::: \n \n{response}")

  def clean_text(self,txt):
    txt = re.sub(r"[^a-zA-Z\s]", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    txt=txt.lower()
    return txt

if __name__ == "__main__":
    loader = TextLoader()