import streamlit as st
import google.generativeai as genai
from ui_components import (
    configure_page,
    render_sidebar,
    initialize_session,
    display_chat_history,
    display_retrieved_documents,
    display_api_key_warning,
)
from chat_handler import handle_chat_input
import os
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.environ.get("GEMINI_API_KEY")

def main():
    """Main application function"""

    configure_page()

    initialize_session()

    retrieval_k = render_sidebar()

    # Configure Gemini
    genai.configure(api_key=gemini_api_key)

    # Initialize RAG
    if st.session_state.rag is None:
        from rag import RetrievalAugmentedGeneration

        st.session_state.rag = RetrievalAugmentedGeneration() # Pake session_state supaya bisa dipakai lintas file

    # Display chat history
    display_chat_history()

    # Chat input dan handle response
    if prompt := st.chat_input("Ketik pertanyaan Anda..."):
        if gemini_api_key:
            handle_chat_input(prompt, retrieval_k)
        else:
            st.error("❌ API Gemini tidak valid, coba cek API terlebih dahulu")


if __name__ == "__main__":
    main()
