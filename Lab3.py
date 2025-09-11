import streamlit as st
from openai import OpenAI
from datetime import datetime

def run():
    st.title("Lab 3c - Interactive Chatbot for Kids 🤖")

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "awaiting_more_info" not in st.session_state:
        st.session_state.awaiting_more_info = False
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""

    # Display previous messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Input box
    if prompt := st.chat_input("Type your message..."):
        # Save the user message always
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if st.session_state.awaiting_more_info:
            # User answered "Do you want more info?"
            if prompt.strip().lower() in ["yes", "y"]:
                more_info_prompt = f"Explain more about this question: '{st.session_state.current_question}' in a way a 10-year-old can understand."

                # Stream assistant reply
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    response = ""
                    today = datetime.now().strftime("%B %d, %Y")
                    system_message = {"role": "system", "content": f"Today's date is {today}. Answer simply for a 10-year-old."}
                    messages_for_openai = [system_message, {"role": "user", "content": more_info_prompt}]
                    stream = client.chat.completions.create(model="gpt-4o-mini", messages=messages_for_openai, stream=True)

                    for chunk in stream:
                        if chunk.choices[0].delta.content:
                            text = chunk.choices[0].delta.content
                            response += text
                            placeholder.markdown(response)

                    st.session_state.messages.append({"role": "assistant", "content": response})

                # Re-ask for more info **every time user says yes**
                st.session_state.awaiting_more_info = True
                st.session_state.messages.append({"role": "assistant", "content": "Do you want more info?"})
                st.chat_message("assistant").write("DO YOU WANT MORE INFO?")
            else:
                # User said no → ask for next question
                st.session_state.awaiting_more_info = False
                st.session_state.current_question = ""
                st.session_state.messages.append({"role": "assistant", "content": "Okay! What question can I help you with next?"})
                st.chat_message("assistant").write("Okay! What question can I help you with next?")
        else:
            # Normal user question
            st.session_state.current_question = prompt  # store current question

            # Stream assistant reply
            with st.chat_message("assistant"):
                placeholder = st.empty()
                response = ""
                today = datetime.now().strftime("%B %d, %Y")
                system_message = {"role": "system", "content": f"Today's date is {today}. Answer simply for a 10-year-old."}
                messages_for_openai = [system_message, {"role": "user", "content": prompt}]
                stream = client.chat.completions.create(model="gpt-4o-mini", messages=messages_for_openai, stream=True)

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        response += text
                        placeholder.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": response})

            # Ask if user wants more info
            st.session_state.awaiting_more_info = True
            st.session_state.messages.append({"role": "assistant", "content": "Do you want more info?"})
            st.chat_message("assistant").write("DO YOU WANT MORE INFO?")

