from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.vectorstores import Chroma, Pinecone, FAISS
from langchain.document_loaders import (
    TextLoader,
    # UnstructuredURLLoader,
    RecursiveUrlLoader,
    PlaywrightURLLoader,
    SeleniumURLLoader,
)
import glob
import os
from os import PathLike
from pathlib import Path
from typing import Union, List, Optional, Literal
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
import pinecone

__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# 참고
# https://python.langchain.com/docs/use_cases/question_answering/how_to/question_answering


def get_pinecone_retriever(
    embedding_model: Embeddings,
    api_key: Optional[str] = None,
    environment: Optional[str] = None,
    index: Optional[str] = None,
):
    pinecone.init(
        api_key=api_key or os.getenv("PINECONE_API_KEY"),
        environment=environment or os.getenv("PINECONE_ENV"),
    )
    vector_db = Pinecone(
        index=pinecone.index.Index(index or os.getenv("PINECONE_INDEX")),
        embedding=embedding_model,
        text_key="text",
    )
    return vector_db.as_retriever()


def get_faiss_ensemble_retriever(
    embedding_model: Embeddings,
    index_path: Union[str, PathLike, Path] = "./law_vectorized_result/",
):
    faiss_db = FAISS.load_local(folder_path=index_path, embeddings=embedding_model)
    return faiss_db.as_retriever()


def get_chroma_retriever(
    collection_name: Literal["law", "precedent"], persist_directory: str = "./chroma"
):
    db = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )
    return db.as_retriever()


if __name__ == "__main__":
    query = "저는 노동자입니다. 사장님이 월급을 안줘요 ㅠㅠ"
    vector_db_chain = get_faiss_ensemble_retriever(OpenAIEmbeddings())

    related_laws = vector_db_chain.get_relevant_documents(query)
    print(related_laws)