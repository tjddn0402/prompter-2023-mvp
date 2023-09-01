from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
from os import PathLike
from typing import Union

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
)


def store_vectors(
    root_path: Union[str, PathLike, bytes] = "./law_data",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                txt_files.append(file_path)

    txt_spliter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedding_model = OpenAIEmbeddings()
    vector_store = Pinecone(
        index=pinecone.Index(os.getenv("PINECONE_INDEX")),
        embedding=embedding_model,
        text_key="text",
    )
    for txt_file in txt_files:
        raw_docs = TextLoader(txt_file).load()
        splited_docs = txt_spliter.split_documents(raw_docs)
        vector_store.add_documents(splited_docs)


if __name__ == "__main__":
    store_vectors()
