import os
import time

from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings #무료, OpenAIEmbeddings  #유료
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS   #무료
from langchain.chains import RetrievalQA

with open("apikey.txt") as f:
    OPENAI_API_KEY = str(f.read())

OPENAI_API_KEY = OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def vectorized_embedding_store() :
    #https://python.langchain.com/docs/integrations/document_loaders/
    #웹페이지에서 가져오는 loader 지정
    # loader = WebBaseLoader()
    loader = TextLoader(r"D:\Gitrepo\prompter-2023-mvp\law_data\saved_data\law_full_save\6ㆍ25전쟁 납북피해 진상규명 및 납북피해자 명예회복에 관한 법률.txt")
    # raw_documents = loader.load()

    #문자열을 vector embbedding하기
    #여기서는 HuggingFace를 사용한다. / #pip install sentence_transformers # HuggingFace Embedding 사용 위해서 필요
    embeddings = HuggingFaceEmbeddings()
    # embeddings = OpenAIEmbeddings()

    #text splitter 설정
    #문서의 양이 많기 때문에 여러개의 서브문서로 분할 하는데 사용
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)

    #vectorstore 설정
    #vectorstore는 Embedding 벡터와 텍스트를 저장하는 DB
    #여기서 사용하는 FAISS는 유사도 검색모델 중 하나로 단어나 문장의 의미가 비슷한 것을 찾을 수 있다.
    #pip install faiss-cpu 설치 필요. FAISS를 사용하기 위해서
    #loader로 읽어들인 데이터를 text_splitter를 통해 분할하고 FAISS를 통해 유사도 검색을 할 수 있도록 vectorstore를 설정
    index = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        text_splitter=text_splitter
        ).from_loaders([loader])

    # 이후 재사용을 위해서 vector db를 파일로 저장
    # 재사용 시에는 embedding과정 필요없음
    #faiss-rus-ukr 폴더가 생성되고 하위에
    #index.faiss, index.pkl 파일이 저장됨
    index.vectorstore.save_local("law_vectorized_result")

def query(text):
    # 전체 데이터를 분할해놓은 서브문서들 중에서 질문과 유사한 내용이 있는 문서들을 찾아내서
    # chat모델에 전달하고 응답을 받음
    chatAI = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1)
    start_time = time.time()
    # https://github.com/langchain-ai/langchain/discussions/4188
    embeddings = HuggingFaceEmbeddings()
    db_call = FAISS.load_local("./law_vectorized_result/", embeddings=embeddings)
    qa_chain = RetrievalQA.from_chain_type(llm=chatAI, retriever=db_call.as_retriever())
    result = qa_chain({"query": text})
    end_time = time.time()
    print(f"Q : {text}\n")
    print(f"A : {result}\n")
    print("time consuming : ", end_time - start_time)
    """
    Q : 6ㆍ25전쟁 납북피해 진상규명 및 납북피해자 명예회복에 관한 법률 시행령의 제개정 이유가 뭐야?

    A : {'query': '6ㆍ25전쟁 납북피해 진상규명 및 납북피해자 명예회복에 관한 법률 시행령의 제개정 이유가 뭐야?', 'result': '제개정이유내용 | [일부개정]◇ 개정이유 및 주요내용 최근 공무상 비밀누설과 관련된 문제의 심각성이 대두되면서 공무원의 비밀유지 의무와 관련된 공직기강을 확립하고, 의무 위반 시 이를 엄중하게 처벌해야 한다는 요구가 커지고 있음. 현행법은 위원회 및 실무위원회의 위원이나 그 직에 있었던 사람이 업무상 알게 된 비밀의 누설에 대하여 징역 2년 이하 또는 1천만원 이하의 벌금에 처하도록 하고 있음. 벌금형은 징역형과 함께 형벌의 대표적 수단으로서 누구나 인정할 수 있는 공정성과 합리성을 지녀야 한다는 점에서 볼 때 위반행위의 불법성에 비례하는 처벌로서 징역형과 벌금형 사이에 균형을 갖출 필요가 있음. 이에 벌금액을 징역 1년당 1천만원의 비율로 조정하여, 직무상 알게 된 비밀을 누설한 사람에게 2년 이하의 징역 또는 2천만원 이하의 벌금에 처하도록 함으로써 업무수행의 공정성과 책임성을 제고하려는 것임.'}
    
    time consuming :  22.410363912582397
    """

# index = vectorized_embedding_store()
query("6ㆍ25전쟁 납북피해 진상규명 및 납북피해자 명예회복에 관한 법률 시행령의 제개정 이유가 뭐야?")

'''
다양한 vector embedding 도구들

__all__ = [
    "OpenAIEmbeddings",
    "HuggingFaceEmbeddings",
    "CohereEmbeddings",
    "ClarifaiEmbeddings",
    "ElasticsearchEmbeddings",
    "JinaEmbeddings",
    "LlamaCppEmbeddings",
    "HuggingFaceHubEmbeddings",
    "MlflowAIGatewayEmbeddings",
    "ModelScopeEmbeddings",
    "TensorflowHubEmbeddings",
    "SagemakerEndpointEmbeddings",
    "HuggingFaceInstructEmbeddings",
    "MosaicMLInstructorEmbeddings",
    "SelfHostedEmbeddings",
    "SelfHostedHuggingFaceEmbeddings",
    "SelfHostedHuggingFaceInstructEmbeddings",
    "FakeEmbeddings",
    "AlephAlphaAsymmetricSemanticEmbedding",
    "AlephAlphaSymmetricSemanticEmbedding",
    "SentenceTransformerEmbeddings",
    "GooglePalmEmbeddings",
    "MiniMaxEmbeddings",
    "VertexAIEmbeddings",
    "BedrockEmbeddings",
    "DeepInfraEmbeddings",
    "DashScopeEmbeddings",
    "EmbaasEmbeddings",
    "OctoAIEmbeddings",
    "SpacyEmbeddings",
    "NLPCloudEmbeddings",
    "GPT4AllEmbeddings",
]
'''

'''
다양한 vector db 들

__all__ = [
    "AlibabaCloudOpenSearch",
    "AlibabaCloudOpenSearchSettings",
    "AnalyticDB",
    "Annoy",
    "AtlasDB",
    "AwaDB",
    "AzureSearch",
    "Cassandra",
    "Chroma",
    "Clickhouse",
    "ClickhouseSettings",
    "DeepLake",
    "DocArrayHnswSearch",
    "DocArrayInMemorySearch",
    "ElasticVectorSearch",
    "ElasticKnnSearch",
    "FAISS",
    "PGEmbedding",
    "Hologres",
    "LanceDB",
    "MatchingEngine",
    "Marqo",
    "Milvus",
    "Zilliz",
    "SingleStoreDB",
    "Chroma",
    "Clarifai",
    "OpenSearchVectorSearch",
    "AtlasDB",
    "DeepLake",
    "Annoy",
    "MongoDBAtlasVectorSearch",
    "MyScale",
    "MyScaleSettings",
    "OpenSearchVectorSearch",
    "Pinecone",
    "Qdrant",
    "Redis",
    "Rockset",
    "SKLearnVectorStore",
    "SingleStoreDB",
    "StarRocks",
    "SupabaseVectorStore",
    "Tair",
    "Tigris",
    "Typesense",
    "Vectara",
    "VectorStore",
    "Weaviate",
    "Zilliz",
    "PGVector",
]

'''
