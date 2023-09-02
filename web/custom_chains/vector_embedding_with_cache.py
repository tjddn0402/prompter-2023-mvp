"""
tiktoken required
"""
import os
import time

from langchain.chat_models import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

with open("apikey.txt") as f:
    OPENAI_API_KEY = str(f.read())

OPENAI_API_KEY = OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def embedding_with_cache():
    underlying_embeddings = OpenAIEmbeddings()

    fs = LocalFileStore("./cache/")

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, fs, namespace=underlying_embeddings.model
    )

    raw_documents = TextLoader(r"D:\Gitrepo\prompter-2023-mvp\law_data\saved_data\law_full_save\6ㆍ25전쟁 납북피해 진상규명 및 납북피해자 명예회복에 관한 법률.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    db = FAISS.from_documents(documents, cached_embedder)

def query(text):
    chatAI = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1)
    start_time = time.time()
    # https://github.com/langchain-ai/langchain/discussions/4188
    underlying_embeddings = OpenAIEmbeddings()
    fs = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, fs, namespace=underlying_embeddings.model
    )
    raw_documents = TextLoader(
        r"D:\Gitrepo\prompter-2023-mvp\law_data\saved_data\law_full_save\6ㆍ25전쟁 납북피해 진상규명 및 납북피해자 명예회복에 관한 법률.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db2 = FAISS.from_documents(documents, cached_embedder)
    qa_chain = RetrievalQA.from_chain_type(llm=chatAI, retriever=db2.as_retriever())
    result = qa_chain({"query": text})
    end_time = time.time()
    print(f"Q : {text}\n")
    print(f"A : {result}\n")
    print("time consuming : ", end_time - start_time)
    """
    Q : 6ㆍ25전쟁 납북피해 진상규명 및 납북피해자 명예회복에 관한 법률 시행령의 제개정 이유가 뭐야?

    A : {'query': '6ㆍ25전쟁 납북피해 진상규명 및 납북피해자 명예회복에 관한 법률 시행령의 제개정 이유가 뭐야?', 'result': '제개정이유내용에 따르면, 개정 이유는 공무상 비밀누설과 관련된 문제의 심각성이 대두되면서 공무원의 비밀유지 의무와 관련된 공직기강을 확립하고, 의무 위반 시 이를 엄중하게 처벌해야 한다는 요구가 커지고 있기 때문입니다. 현행법은 위원회 및 실무위원회의 위원이나 그 직에 있었던 사람이 업무상 알게 된 비밀의 누설에 대하여 징역 2년 이하 또는 1천만원 이하의 벌금에 처하도록 하고 있습니다. 이에 벌금액을 징역 1년당 1천만원의 비율로 조정하여, 직무상 알게 된 비밀을 누설한 사람에게 2년 이하의 징역 또는 2천만원 이하의 벌금에 처하도록 함으로써 업무수행의 공정성과 책임성을 제고하려는 것입니다.'}
    
    time consuming :  13.363122940063477
    """

embedding_with_cache()
query("6ㆍ25전쟁 납북피해 진상규명 및 납북피해자 명예회복에 관한 법률 시행령의 제개정 이유가 뭐야?")