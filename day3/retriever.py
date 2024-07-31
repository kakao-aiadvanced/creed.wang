from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Retriever:
    def __init__(self, urls):
        self.urls = urls

    def create(self):
        docs = [WebBaseLoader(url).load() for url in self.urls]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vectorDB
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        )
        return vectorstore.as_retriever()
