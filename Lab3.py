import streamlit as st
from openai import OpenAI
from datetime import datetime

def run():
    st.title("Lab 3 - Streaming Chatbot 🤖")

    # Initialize OpenAI client with API key
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Keep chat history in session
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous messages
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Input box
    if prompt := st.chat_input("Type your message..."):
        # Add user message to session
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # --- Conversation buffer: last 2 user messages + their assistant responses ---
        buffer_messages = []
        user_count = 0
        for msg in reversed(st.session_state["messages"]):
            if msg["role"] == "user":
                user_count += 1
            if user_count <= 2:
                buffer_messages.insert(0, msg)  # prepend to keep order
            if user_count == 2 and msg["role"] == "user":
                break

        # --- Add system message with real-time date ---
        today = datetime.now().strftime("%B %d, %Y")
        system_message = {
            "role": "system",
            "content": f"You are a helpful assistant. Today's date is {today}."
        }
        messages_for_openai = [system_message] + buffer_messages

        # Stream assistant reply
        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = ""
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_for_openai,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    response += text
                    placeholder.markdown(response)  # update in same block

        # Save assistant response
        st.session_state["messages"].append(
            {"role": "assistant", "content": response}
        )


