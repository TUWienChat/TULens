"""
Streamlit GUI for TU Wien RAG System
"""

import streamlit as st
from rag_system import TUWienRAG
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="TU Wien Informatics Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .answer-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
    }
    .stats-card {
        background-color: #e8f4ea;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.initialized = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "response_language" not in st.session_state:
    st.session_state.response_language = "English"

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### 🌐 Response Language")
    st.session_state.response_language = st.radio(
        "Select language",
        options=["English", "German"],
        index=0 if st.session_state.response_language == "English" else 1,
        horizontal=True,
        help="Choose the language for responses"
    )

    st.divider()

    st.markdown("### About")
    st.info(
        "This is a RAG-powered assistant for TU Wien informatics master programs. "
        "It uses scraped web pages and curriculum PDFs to answer your questions."
    )
    
    st.divider()
    
    st.markdown("### 📄 Add Documents")
    st.caption("Upload PDFs to add context for this session. These are stored temporarily in memory.")
    
    uploaded_file = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        help="Upload PDF files to add to the session knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("📥 Add PDF to Session", key="add_pdf_btn"):
            if st.session_state.rag_system is not None and st.session_state.initialized:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        pdf_bytes = uploaded_file.read()
                        chunks_added = st.session_state.rag_system.add_pdf_from_bytes(pdf_bytes, uploaded_file.name)
                        st.success(f"✅ Added {chunks_added} chunks from {uploaded_file.name}!")
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")
            else:
                st.warning("RAG system not initialized yet. Please wait.")
    
    # Show user document count
    if st.session_state.initialized and st.session_state.rag_system:
        user_doc_count = st.session_state.rag_system.get_user_document_count()
        if user_doc_count > 0:
            st.markdown(f"**📊 Session documents:** {user_doc_count} chunks")
            if st.button("🗑️ Clear Uploaded Documents"):
                st.session_state.rag_system.clear_user_documents()
                st.success("Cleared uploaded documents!")
                st.rerun()
    
    st.divider()
    
    # Reset conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    
# Main content
st.title("🎓 TU Wien Informatics Assistant")
st.markdown("Ask questions about TU Wien master programs, curricula, courses, and admissions")

# Initialize RAG system
if not st.session_state.initialized:
    with st.spinner("🔄 Connecting to knowledge base..."):
        try:
            # Check for API key
            if not os.getenv("GROQ_API_KEY"):
                st.error("❌ GROQ_API_KEY not found in environment variables")
                st.info("Please set GROQ_API_KEY in your .env file")
                st.stop()
            
            # Initialize RAG system (connects to existing ChromaDB)
            st.session_state.rag_system = TUWienRAG()
            st.session_state.rag_system.initialize()
            st.session_state.initialized = True
            st.success("✅ Connected to TU Wien knowledge base!")
        except Exception as e:
            st.error(f"❌ Error initializing RAG system: {str(e)}")
            st.info("Make sure ChromaDB is running (docker-compose up) and documents are ingested")
            st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 View sources"):
                for i, source in enumerate(message["sources"], 1):
                    source_type = source.metadata.get('source_type', '')
                    doc_type = source.metadata.get('type', '')
                    
                    st.markdown(f"**Source {i}:**")
                    
                    if source_type == 'user_upload':
                        # User uploaded PDF
                        pdf_name = source.metadata.get('pdf_name', 'Unknown')
                        page = source.metadata.get('page', 'N/A')
                        st.markdown(f"⭐ Uploaded: {pdf_name} (Page {page})")
                    elif doc_type == 'pdf':
                        # Pre-scraped PDF with rich metadata
                        title = source.metadata.get('title', '')
                        program = source.metadata.get('program', '')
                        page = source.metadata.get('page', 'N/A')
                        pdf_url = source.metadata.get('pdf_url', '')
                        
                        display_name = title if title else (pdf_url.split('/')[-1] if pdf_url else 'Unknown PDF')
                        st.markdown(f"📄 {display_name} (Page {page})")
                        if program:
                            st.markdown(f"📌 Program: {program}")
                        if pdf_url:
                            st.markdown(f"🔗 [View PDF]({pdf_url})")
                    else:
                        # Web source
                        source_url = source.metadata.get('source', 'Unknown')
                        program = source.metadata.get('program', '')
                        st.markdown(f"🌐 [{source_url}]({source_url})")
                        if program:
                            st.markdown(f"📌 Program: {program}")
                    
                    st.markdown(f"> {source.page_content[:250]}...")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask about TU Wien informatics programs..."):
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                # Build conversation history
                conversation_history = [msg for msg in st.session_state.messages[:-1]]
                
                # Query with conversation context and language preference
                answer, sources = st.session_state.rag_system.query(
                    prompt, 
                    conversation_history=conversation_history,
                    response_language=st.session_state.response_language
                )
                st.markdown(answer)
                
                # Display sources
                if sources:
                    st.divider()
                    with st.expander(f"📚 Sources ({len(sources)} references)", expanded=False):
                        # Group by source type: user uploads vs pre-scraped
                        user_sources = []
                        web_sources = []
                        pdf_sources = []
                        
                        for source in sources:
                            source_type = source.metadata.get('source_type', '')
                            if source_type == 'user_upload':
                                user_sources.append(source)
                            elif source.metadata.get('type') == 'pdf':
                                pdf_sources.append(source)
                            else:
                                web_sources.append(source)
                        
                        # Show user-uploaded sources first (highest priority)
                        if user_sources:
                            st.markdown("### ⭐ Your Uploaded Documents")
                            for i, doc in enumerate(user_sources, 1):
                                pdf_name = doc.metadata.get('pdf_name', 'Unknown')
                                page = doc.metadata.get('page', 'N/A')
                                st.markdown(f"**{i}.** 📄 {pdf_name} (Page {page})")
                                st.markdown(f"> {doc.page_content[:200]}...")
                                st.divider()
                        
                        if web_sources:
                            st.markdown("### 🌐 Web Sources")
                            for i, doc in enumerate(web_sources, 1):
                                source_url = doc.metadata.get('source', 'Unknown')
                                program = doc.metadata.get('program', '')
                                st.markdown(f"**{i}.** [{source_url}]({source_url})")
                                if program:
                                    st.markdown(f"   Program: {program}")
                                st.markdown(f"> {doc.page_content[:200]}...")
                                st.divider()
                        
                        if pdf_sources:
                            st.markdown("### 📄 PDF Sources")
                            for i, doc in enumerate(pdf_sources, 1):
                                # Get rich metadata from extract_pdf
                                title = doc.metadata.get('title', '')
                                program = doc.metadata.get('program', '')
                                page = doc.metadata.get('page', 'N/A')
                                pdf_url = doc.metadata.get('pdf_url', '')
                                parent_url = doc.metadata.get('parent_url', '')
                                
                                # Display title or fallback to PDF URL filename
                                if title:
                                    display_name = title
                                elif pdf_url:
                                    display_name = pdf_url.split('/')[-1]
                                else:
                                    display_name = 'Unknown PDF'
                                
                                st.markdown(f"**{i}.** 📄 {display_name} (Page {page})")
                                if program:
                                    st.markdown(f"   📌 Program: {program}")
                                if pdf_url:
                                    st.markdown(f"   🔗 [View PDF]({pdf_url})")
                                st.markdown(f"> {doc.page_content[:200]}...")
                                st.divider()
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.85rem;">
    <p>TU Wien Informatics RAG System | Powered by Groq, LangChain & ChromaDB</p>
    </div>
""", unsafe_allow_html=True)
