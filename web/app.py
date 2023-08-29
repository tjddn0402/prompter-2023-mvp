# help from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
# help from https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py
from typing import Any
import streamlit as st

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from custom_chains.translation import (
    get_answer_translation_chain,
    get_inquiry_translation_chain,
)
from custom_chains.qa import getVectorDB
from custom_chains.chat import get_legal_help_chain


class LegalChatbot:
    def __init__(self):
        self.llm_translation = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.llm_advisor = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
        self.embedding_model = OpenAIEmbeddings()
        self.inquiry_translation_chain = get_inquiry_translation_chain(self.llm_translation)
        self.db = getVectorDB(self.embedding_model)
        self.legal_help_chain = get_legal_help_chain(self.llm_advisor)
        self.answer_translation_chain = get_answer_translation_chain(self.llm_translation)

    def __call__(self, visitor_type, eng_query):
        eng_query = f"I visiting South Korea for {visitor_type}. " + eng_query
        kor_inquiry = self.inquiry_translation_chain.run(eng_query)
        related_laws = self.db.get_relevant_documents(kor_inquiry)
        kor_advice = self.legal_help_chain.run(
            inquiry=kor_inquiry, related_laws=related_laws
        )
        eng_advice = self.answer_translation_chain.run(
            tgt_lang="English", legal_help=kor_advice
        )
        return eng_advice


chatbot = LegalChatbot()


def main():
    st.set_page_config(page_title="Korea Law Help", page_icon="⚖️")
    st.title("Do you need any help about Korean Laws?")

    visitor_type = st.selectbox(
        "I visit South Korea for ...",
        ("Select", "Tour", "Study", "Work", "immigration", "etc"),
    )
    if visitor_type == "Select":
        visitor_type = None
    elif visitor_type == "etc":
        visitor_type = st.text_input("What is your purpose to visit South Korea?")

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

        response = chatbot(visitor_type, query)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
