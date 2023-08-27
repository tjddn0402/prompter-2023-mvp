# help from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
import streamlit as st


def Chatbot(query):
    """langchain에서 받아온 답변으로 대체할 예정."""
    return "wait.. i'm preparing to help you"


def main():
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

        response = Chatbot(query)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
