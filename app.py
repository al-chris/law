import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import json
from rag import global_stream_response

st.set_page_config(
    page_title="Law",
    page_icon=":material/robot:",
    initial_sidebar_state="collapsed"
)

# Set up session state to store the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.title("Gemma chatbot")
    if st.button("Clear chat :material/delete:", type="primary", use_container_width=True):
        st.session_state.messages = []



messages = st.container(height=410)

with messages:
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)



if prompt := st.chat_input("Say something", max_chars=350):
    # Add user message to the session state
    st.session_state.messages.append(HumanMessage(content=prompt))

    # print(st.session_state.messages)
    
    # Simple bot response
    # bot_message = create_response(json.dumps(st.session_state.messages))
    

    # For streaming the response
    with messages:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
                # bot_message = st.write_stream(stream_response(st.session_state.messages))
                bot_message = st.write_stream(global_stream_response(prompt))

    st.session_state.messages.append(AIMessage(content=bot_message))

    st.rerun()

