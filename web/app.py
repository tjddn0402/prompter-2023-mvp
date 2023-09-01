# help from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
# help from https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py
import streamlit as st
from jinja2 import Template
import pinecone
import json
import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ChatMessageHistory,
)
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from typing import Any, Dict
from dotenv import load_dotenv

from custom_chains.translation import (
    get_answer_translation_chain,
    get_inquiry_translation_chain,
)
from custom_chains.retriever import get_chroma_retriever, get_pinecone_retriever
from custom_chains.chat import get_legal_help_chain


load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
)


class LegalChatbot:
    def __init__(self, verbose: bool = False):
        self.memory = ConversationSummaryMemory(
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
        )
        self.prev_summary = ""

        self.llm_translator = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.llm_advisor = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
        self.embedding_model = OpenAIEmbeddings()

        self.inquiry_translation_chain = get_inquiry_translation_chain(
            self.llm_translator
        )
        self.retriever = get_pinecone_retriever(self.embedding_model)
        self.legal_help_chain = get_legal_help_chain(self.llm_advisor)
        self.answer_translation_chain = get_answer_translation_chain(
            self.llm_translator
        )

        self.verbose = verbose

    def __call__(self, eng_query: str) -> Dict:
        kor_inquiry = self.inquiry_translation_chain.run(eng_query)
        kor_inquiry_dict = json.loads(kor_inquiry)
        related_laws = self.retriever.get_relevant_documents(
            kor_inquiry_dict["inquiry"]
        )
        kor_advice = self.legal_help_chain.run(
            inquiry=kor_inquiry, related_laws=related_laws, history=self.prev_summary
        )
        eng_advice = self.answer_translation_chain.run(
            tgt_lang="english", legal_help=kor_advice
        )
        eng_advice_dict = json.loads(eng_advice)
        eng_advice_format = """{{ conclusion }}
        
# AI Lawyer's advice
{{ advice }}

# Related Laws
{% for law in related_laws %}
- {{law}}
{% endfor %}
"""
        answer_template = Template(eng_advice_format)

        eng_answer = answer_template.render(
            conclution=eng_advice_dict["conclusion"],
            related_laws=eng_advice_dict["related_laws"],
            advice=eng_advice_dict["advice"],
        )

        # 대화 내용 요약 후 메모리에 저장
        self.memory.save_context({"client": eng_query}, {"lawyer": eng_answer})
        self.prev_summary = self.memory.predict_new_summary(
            messages=self.memory.chat_memory.messages,
            existing_summary=self.prev_summary,
        )

        if self.verbose:
            # print("english query:", eng_query)
            # print("korean query:", kor_inquiry_dict)
            # print("retrieved passages:", related_laws)
            # print("advice (korean):", kor_advice)
            # print("advice (english):", eng_advice_dict)
            # print()
            print("memory")
            print(self.memory.chat_memory.messages)
            print()
            print("summary")
            print(self.prev_summary)
            print()

        return eng_answer


chatbot = LegalChatbot(verbose=True)


def main():
    st.set_page_config(page_title="Korea Law Help", page_icon="⚖️")
    st.title("Do you need any help about Korean Laws?")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    print(st.session_state)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if query := st.chat_input(
        "Please explain about your situation. Are you student? worker? tourist?"
    ):
        # Display user message in chat message container
        st.chat_message("user").markdown(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        response = chatbot(query)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
