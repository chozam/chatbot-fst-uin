import streamlit as st
import google.generativeai as genai
from ui_components import display_retrieved_documents


def build_context(retrieved_docs):
    """Build context dari retrieved documents"""

    if retrieved_docs:
        context = "\n".join([
            f"- {doc['content'][:300]}..." 
            for doc in retrieved_docs
        ])
    else:
        context = "Tidak ada dokumen yang relevan ditemukan."
    
    return context


def create_user_message(prompt, context):
    """Create formatted user message dengan context"""

    user_message = f"""Konteks dari dokumen:
    {context}

    ---

    Pertanyaan: {prompt}

    Berikan jawaban yang informatif berdasarkan konteks di atas."""
    
    return user_message


def get_system_prompt():
    """Get system prompt untuk AI"""

    return """Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan.
Gunakan konteks dari dokumen untuk memberikan jawaban yang akurat dan relevan.
Jika informasi tidak tersedia di konteks, sampaikan dengan jelas bahwa Anda tidak menemukan informasi tersebut.
Berikan jawaban dalam bahasa Indonesia yang jelas dan mudah dipahami."""


def generate_gemini_response(user_message):
    """Generate response dari Gemini API"""

    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(
        user_message,
        safety_settings=[
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
    )
    
    return response.text


def handle_chat_input(prompt, retrieval_k):
    """Handle user input dan generate response"""

    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🤔 Sedang memproses..."):
            try:
                # Retrieve documents dari RAG
                retrieved_docs = st.session_state.rag.manual_retrieval_documents(
                    prompt, 
                    k=retrieval_k
                )
                
                # Build context dari retrieved documents
                context = build_context(retrieved_docs)
                
                # Create user message dengan context
                user_message = create_user_message(prompt, context)
                
                answer = generate_gemini_response(user_message)
   
                display_retrieved_documents(retrieved_docs)
                
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")