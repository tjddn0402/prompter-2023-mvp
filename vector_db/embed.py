from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.vectorstores import Pinecone, Chroma, LanceDB
from langchain.document_loaders import (
    TextLoader,
    # UnstructuredURLLoader,
    RecursiveUrlLoader,
    PlaywrightURLLoader,
    SeleniumURLLoader,
)
import glob
import os
from typing import Union, List

__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


def getVectorDB(
    root: Union[str, bytes, os.PathLike] = "vector_db/laws/*/*.txt",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
):
    txt_spliter = TokenTextSplitter(
        model_name="text-embedding-ada-002",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embedding_model = OpenAIEmbeddings()
    vector_db = Chroma(embedding_function=embedding_model)
    txts = sorted(glob.glob(root))
    for txt_path in txts:
        raw_doc = TextLoader(txt_path).load()
        splited_docs = txt_spliter.split_documents(raw_doc)
        vector_db.add_documents(splited_docs)
    return vector_db


if __name__ == "__main__":
    law_db = getVectorDB()
    similar_laws = law_db.similarity_search("사장님이 월급을 안줘요", k=3)
    print(similar_laws)
