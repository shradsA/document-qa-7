import streamlit as st
from openai import OpenAI

def run():
    st.title("Lab 3 - Streaming Chatbot 🤖")

    # Initialize OpenAI client with API key
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Keep chat history in session
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful chatbot."}
        ]

    # Display previous messages
    for msg in st.session_state["messages"]:
        if msg["role"] != "system":
            st.chat_message(msg["role"]).write(msg["content"])

    # Input box
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Stream assistant reply
        with st.chat_message("assistant"):
            placeholder = st.empty()  # placeholder to update text in place
            response = ""
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state["messages"],
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    response += text
                    placeholder.markdown(response)  # update in the same block

            # Save assistant response
            st.session_state["messages"].append(
                {"role": "assistant", "content": response}
            )
