import streamlit as st

st.set_page_config(page_title="Multi-page Labs App", page_icon="📚", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a Lab:",
    [
        "Lab 1",
        "Lab 2 (Default)",
        "Lab 3 (Chatbot)",
        "Lab 4 (VectorDB)",
        "Lab 5 (What to Wear Bot)",
        "Lab 6 (AI Fact-Checker + Citation Builder)",
        "Lab 8 (RAG + Reranking 10-Q)"
    ]
)

if page == "Lab 1":
    import Lab1
    Lab1.run()
elif page == "Lab 2 (Default)":
    import Lab2
    Lab2.run()
elif page == "Lab 3 (Chatbot)":
    import Lab3
    Lab3.run()
elif page == "Lab 4 (VectorDB)":
    import lab4
    lab4.main()
elif page == "Lab 5 (What to Wear Bot)":
    import lab5
    lab5.run()
elif page == "Lab 6 (AI Fact-Checker + Citation Builder)":
    import lab6
    lab6.run()
elif page == "Lab 8 (RAG + Reranking 10-Q)":
    import lab8  # lowercase matches filename
    lab8.run()
