import os
import requests
import streamlit as st

CHAT_HOST = os.environ.get('CHAT_HOST', 'fastapi')

def responder(prompt):
    url = 'http://aa703202c6c1849ce9814d2c1fbe6a9c-697758537.ca-central-1.elb.amazonaws.com'
    url = f'http://{CHAT_HOST}'

    response = requests.post(f"{url}/respond",
                         headers={"Content-Type": "application/json",
                                  "Accept": "text/event-stream"},
                         json={"content": prompt},
                         stream=True)

    for chunk in response:
        yield chunk.decode("utf-8")


st.set_page_config(page_title='Eberron DM Assistant', layout='wide', initial_sidebar_state='auto')
st.title("Eberron DM Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = responder(prompt)
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
