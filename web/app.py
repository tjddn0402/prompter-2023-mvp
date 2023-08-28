# help from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
# help from https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate


def Chatbot(visitor_type, query):
    query = f"I visited South Korea for {visitor_type}. " + query
    return (
        f"OK, you visit South Korea for {visitor_type}, and your inquiry is '{query}'!"
    )


def main():
    st.set_page_config(page_title="Korea Law Help", page_icon="⚖️")
    st.title("Do you need any help about Korean Laws?")

    visitor_type = st.selectbox(
        "I visit South Korea for ...",
        ("Tour", "Study", "Work", "immigration", "etc"),
    )
    if visitor_type == "etc":
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

        response = Chatbot(visitor_type, query)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
