import streamlit as st
import os
import PyPDF2
from io import BytesIO
from typing import List

# SQLite fix for Streamlit Cloud / ChromaDB
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# ChromaDB import
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from openai import OpenAI

# ---------------- PDF Utilities ----------------
def read_pdf_content(pdf_file) -> str:
    """Extract text from PDF"""
    try:
        pdf_file.seek(0)
        reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# ---------------- Vector DB ----------------
def create_vector_database():
    """Initialize ChromaDB and OpenAI client"""
    if not CHROMADB_AVAILABLE:
        st.error("ChromaDB not installed. Run: pip install chromadb pysqlite3-binary")
        return None

    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key not found.")
        return None

    openai_client = OpenAI(api_key=openai_api_key)
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(
        name="Lab4Collection",
        metadata={"description": "PDF documents for Lab4 vector search"}
    )
    return {"client": chroma_client, "collection": collection, "openai_client": openai_client}

def process_uploaded_pdfs(vector_db, uploaded_files: List):
    """Process PDF uploads into vector DB"""
    if not uploaded_files:
        st.warning("No PDF files uploaded")
        return False

    if len(uploaded_files) > 7:
        st.warning("Max 7 PDFs. Only first 7 will be used.")
        uploaded_files = uploaded_files[:7]

    collection = vector_db["collection"]
    client = vector_db["openai_client"]

    documents, metadatas, ids = [], [], []

    progress = st.progress(0)
    status = st.empty()

    for idx, f in enumerate(uploaded_files):
        status.text(f"Processing {f.name}...")
        text = read_pdf_content(f)
        if text:
            documents.append(text[:30000])
            metadatas.append({"filename": f.name})
            ids.append(f"doc_{idx}_{f.name.replace(' ', '_')}")
        progress.progress((idx + 1)/len(uploaded_files))

    if not documents:
        st.warning("No valid text extracted from PDFs")
        return False

    # Generate embeddings
    embeddings = []
    for doc in documents:
        emb = client.embeddings.create(input=doc, model="text-embedding-3-small")
        embeddings.append(emb.data[0].embedding)

    collection.add(documents=documents, metadatas=metadatas, embeddings=embeddings, ids=ids)
    st.success(f"{len(documents)} PDFs processed into vector DB")
    return True

def search_vector_database(vector_db, query: str, top_k: int = 3):
    """Retrieve top K relevant documents"""
    collection = vector_db["collection"]
    client = vector_db["openai_client"]

    emb = client.embeddings.create(input=query, model="text-embedding-3-small")
    query_emb = emb.data[0].embedding

    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    return results

# ---------------- RAG Prompt ----------------
def create_rag_prompt(user_question: str, context_docs: List[str], source_files: List[str]) -> str:
    context_text = "\n\n".join([f"Document {i+1}:\n{doc[:2000]}" for i, doc in enumerate(context_docs)])
    sources = ", ".join(source_files)
    prompt = f"""
You are a helpful course assistant. Use the context from these documents: {sources}

CONTEXT:
{context_text}

QUESTION:
{user_question}

Answer using the context. If context is insufficient, clearly mention it and use general knowledge.
"""
    return prompt

# ---------------- Main Streamlit App ----------------
def main():
    st.set_page_config(page_title="Lab 4 RAG Chatbot", layout="wide")
    st.title("📚 Course Information Chatbot (RAG)")

    if not CHROMADB_AVAILABLE:
        st.error("ChromaDB is not installed. Run: pip install chromadb pysqlite3-binary")
        return

    # Initialize vector DB once per session
    if "vector_db" not in st.session_state:
        st.info("Initializing vector database...")
        st.session_state.vector_db = create_vector_database()

    vector_db = st.session_state.vector_db
    if not vector_db:
        st.error("Vector DB not initialized.")
        return

    # Sidebar - Upload PDFs
    st.sidebar.header("📄 PDF Management")
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if st.sidebar.button("Process PDFs"):
        if uploaded_files:
            process_uploaded_pdfs(vector_db, uploaded_files)
        else:
            st.sidebar.warning("Upload PDFs first")

    # Sidebar - Clear database
    if st.sidebar.button("Clear Database"):
        try:
            collection = vector_db["collection"]
            all_docs = collection.get()
            if all_docs["ids"]:
                collection.delete(ids=all_docs["ids"])
            st.success("Vector database cleared")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error clearing DB: {e}")

    # Chatbot interface
    st.header("💬 Ask your course questions")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                st.caption(f"Sources: {msg['sources']}")

    # Chat input
    if prompt := st.chat_input("Ask me about your course materials..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                try:
                    results = search_vector_database(vector_db, prompt)
                    if results and results["documents"]:
                        context_docs = results["documents"][0]
                        source_files = [meta["filename"] for meta in results["metadatas"][0]]
                        rag_prompt = create_rag_prompt(prompt, context_docs, source_files)
                        client = vector_db["openai_client"]
                        resp = client.chat.completions.create(
                            model="gpt-5-mini",
                            messages=[{"role": "user", "content": rag_prompt}],
                            temperature=0.2
                        )
                        answer = resp.choices[0].message.content
                        st.markdown(answer)
                        st.caption(f"Sources: {', '.join(source_files)}")
                        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": ", ".join(source_files)})
                    else:
                        fallback = "No relevant info found in uploaded PDFs."
                        st.markdown(fallback)
                        st.session_state.chat_history.append({"role": "assistant", "content": fallback})
                except Exception as e:
                    st.error(f"Error generating response: {e}")

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# ---------------- Run App ----------------
if __name__ == "__main__":
    main()







