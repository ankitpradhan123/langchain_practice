import streamlit as st
import constants
import os
from rag import Rag

# Streamlit UI
st.title("📄 RAG Chat App with PDF Upload & Chat History")

# Sidebar for uploaded PDFs
st.sidebar.title("Uploaded PDFs")
uploaded_files = os.listdir(constants.UPLOAD_FOLDER)

# Chat history storage
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_files:
    selected_file = st.sidebar.selectbox("📄 Select a file to remove", uploaded_files)
    if st.sidebar.button("❌ Remove Selected PDF"):
        file_path = os.path.join(constants.UPLOAD_FOLDER, selected_file)
        os.remove(file_path)  # Remove file from storage
        st.success(f"📂 File '{selected_file}' removed!")

        # If no PDFs remain, delete FAISS index
        if not os.listdir(constants.UPLOAD_FOLDER):
            #Remove FAISS index
            if os.path.exists(constants.FAISS_INDEX_FILE):
                os.remove(constants.FAISS_INDEX_FILE)
            if os.path.exists(constants.FAISS_INDEX):
                os.remove(constants.FAISS_INDEX)
            st.success("🗑️ FAISS index deleted as no PDFs remain!")

        # Refresh page
        st.rerun()

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file and uploaded_file not in uploaded_files:
    file_path = os.path.join(constants.UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"📂 File '{uploaded_file.name}' uploaded successfully!")
    rag = Rag(file_path, uploaded_file.name)
    vectorstore = rag.process_pdf()
    st.success("✅ FAISS index updated!")

# Load FAISS index
vectorstore = Rag.load_vector()
if not vectorstore:
    st.warning("⚠️ No FAISS index found. Please upload a PDF first.")

    # Chat Interface
query = st.text_input("🗨️ Ask a question based on the uploaded document")
if query and vectorstore:
    context, sources = Rag.retrieve_context(vectorstore, query)
    answer = Rag.generate_response(query, context)

    # Save chat to history
    st.session_state.chat_history.append({"query": query, "answer": answer, "sources": sources})

    # Display results
    st.write("### 🧠 Answer:")
    st.write(answer)

    # Show sources
    st.write("#### 📜 Sources:")
    for doc in sources:
        st.markdown(f"- **Source**: {doc.page_content}")

# Load selected chat history if clicked
if "selected_chat" in st.session_state:
    st.write("### 🔄 Previous Conversation")
    st.write(f"**Question:** {st.session_state.selected_chat['query']}")
    st.write(f"**Answer:** {st.session_state.selected_chat['answer']}")
    st.write("#### 📜 Sources:")
    for doc in st.session_state.selected_chat["sources"]:
        st.markdown(f"- **Source**: {doc.page_content}")