import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env

def run():
    st.title("Lab 1 - Document Q&A")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found! Please add it to your .env file.")
        return

    client = OpenAI(api_key=openai_api_key)

    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file
    )

    if uploaded_file and question:
        document = uploaded_file.read().decode()
        messages = [
            {"role": "user", "content": f"Here's a document: {document}\n\n---\n\n{question}"}
        ]
        stream = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            stream=True,
        )
        st.write_stream(stream)

