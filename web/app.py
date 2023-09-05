# help from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
# help from https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py
import streamlit as st

from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()

from legal_chatbot import LegalChatbot

chatbot = LegalChatbot(verbose=True)


def main():
    st.set_page_config(page_title="LexServo", page_icon="⚖️")
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
        "What kind of legal help do you need? Explain your situation."
    ):
        # Display user message in chat message container
        st.chat_message("user").markdown(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Wait for it... It might take a minute"):
            response = chatbot(query)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
