# help from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
# help from https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py
from jinja2 import Template
import json
import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ChatMessageHistory,
)
from langchain.schema.document import Document

from typing import Any, Dict, List
from dotenv import load_dotenv

from custom_chains.retriever import (
    get_chroma_retriever,
    get_pinecone_retriever,
    get_faiss_ensemble_retriever,
)
from custom_chains.chat import get_legal_help_chain


load_dotenv()


class LegalChatbot:
    def __init__(self, verbose: bool = False):
        self.memory = ConversationSummaryMemory(
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
        )
        self.prev_summary = "Dialog is not started yet."

        self.llm_advisor = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
        self.embedding_model = OpenAIEmbeddings()

        # self.retriever = get_faiss_ensemble_retriever(self.embedding_model)
        self.retriever = get_pinecone_retriever(self.embedding_model)
        self.legal_help_chain = get_legal_help_chain(self.llm_advisor)

        self.verbose = verbose

    def __proc_legal_doc(self, doc:Document):
        src, _ = os.path.splitext(os.path.basename(doc.metadata["source"]))
        return "#"+src+'\n'+doc.page_content

    def __call__(self, inquiry: str) -> Dict:
        legal_basis:List[Document] = self.retriever.get_relevant_documents(inquiry)
        legal_basis:List[str] = list(map(self.__proc_legal_doc, legal_basis))
        legal_basis = "\n\n\n".join(legal_basis)

        eng_advice = self.legal_help_chain.run(
            inquiry=inquiry, related_laws=legal_basis, history=self.prev_summary
        )
        eng_advice_dict = json.loads(eng_advice)
        eng_advice_format = """{{ conclusion }}
        
# Legal Solution
{{ advice }}

## Legal Basis
{% for law in related_laws %}
- {{law}}
{% endfor %}
"""
        answer_template = Template(eng_advice_format)
        rendered_answer = answer_template.render(
            conclusion=eng_advice_dict["conclusion"],
            advice=eng_advice_dict["legal_solution"],
            related_laws=eng_advice_dict["legal_basis"],
        )

        # 대화 내용 요약 후 메모리에 저장
        self.memory.save_context(
            {"client": inquiry}, {"lawyer": eng_advice_dict["conclusion"]}
        )
        self.prev_summary = self.memory.predict_new_summary(
            messages=self.memory.chat_memory.messages,
            existing_summary=self.prev_summary,
        )

        if self.verbose:
            print("#" * 100)
            print("legal basis")
            print(legal_basis)
            print()
            print("memory")
            for msg in self.memory.chat_memory.messages:
                print(msg)
            print()
            print("summary")
            print(self.prev_summary)
            print("#" * 100)

        return rendered_answer


if __name__ == "__main__":
    chatbot = LegalChatbot(verbose=True)
    while True:
        query = input("ask something about law: ")
        if query == "q":
            break
        answer = chatbot(query)
        print(answer)
