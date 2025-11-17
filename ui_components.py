import streamlit as st
import os
from rag import RetrievalAugmentedGeneration


def configure_page():
    """Konfigurasi halaman Streamlit"""
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🤖 Chatbot FST UIN SuKa")


def render_sidebar():
    """Render sidebar dengan konfigurasi"""
    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        
        st.subheader("Pengaturan RAG")
        retrieval_k = st.slider("Jumlah dokumen yang diambil", 1, 10, 5)
        
        # Upload PDF untuk menambah knowledge base
        st.subheader("📄 Tambah Dokumen")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file and st.button("Upload ke Knowledge Base"):
            handle_file_upload(uploaded_file)
        
        if st.button("🗑️ Hapus Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        return retrieval_k


def handle_file_upload(uploaded_file):
    """Handle upload file PDF ke knowledge base"""
    with st.spinner("Mengupload dokumen..."):
        try:
            # Simpan file sementara
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Initialize RAG dan upload
            rag = RetrievalAugmentedGeneration()
            rag.load_to_supabase(temp_path)
            
            # Hapus file sementara
            os.remove(temp_path)
            st.success("✅ Dokumen berhasil diupload!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def initialize_session():
    """Inisialisasi session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = None


def display_chat_history():
    """Tampilkan chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def display_retrieved_documents(retrieved_docs):
    """Tampilkan dokumen yang diambil"""
    with st.expander("📚 Dokumen yang Diambil"):
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs, 1):
                st.write(f"**Dokumen {i}** (Similarity: {doc['similarity']:.2f})")
                st.write(doc['content'][:500] + "...")
        else:
            st.write("Tidak ada dokumen yang diambil.")


def display_api_key_warning():
    """Tampilkan warning jika API key belum dimasukkan"""
    st.warning("⚠️ Silakan masukkan Gemini API Key di sidebar")