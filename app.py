# app.py
"""Main Streamlit application for ChaiCode Documentation Chatbot"""

import streamlit as st
import os
import time
from typing import Dict, List, Any

# Import our custom modules
from config import Config, EmbeddingManager
from ingest_docs import ChaiCodeDocsCrawler, convert_to_langchain_docs
from chunks import DocumentChunker, optimize_chunks_for_retrieval
from query import ChaiCodeQueryProcessor, create_system_prompts

# Page configuration
st.set_page_config(
    page_title="ChaiCode Docs Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    
    # Core system components
    if 'embedding_manager' not in st.session_state:
        st.session_state.embedding_manager = None
    
    if 'query_processor' not in st.session_state:
        st.session_state.query_processor = None
    
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    
    # System status
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    
    if 'docs_loaded' not in st.session_state:
        st.session_state.docs_loaded = False
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Statistics
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'documents_processed': 0,
            'chunks_created': 0
        }

def validate_api_keys() -> tuple[bool, str]:
    """Validate that required API keys are present"""
    
    if not Config.OPENAI_API_KEY:
        return False, "OpenAI API key is missing. Please check your environment variables or enter it in the sidebar."
    
    if not Config.PINECONE_API_KEY:
        return False, "Pinecone API key is missing. Please check your environment variables or enter it in the sidebar."
    
    return True, "API keys validated successfully!"

def setup_sidebar():
    """Setup the sidebar with configuration options"""
    
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # API Keys section
        st.subheader("🔐 API Keys")
        
        # Allow users to input API keys if not in environment
        openai_key = st.text_input(
            "OpenAI API Key",
            value=Config.OPENAI_API_KEY or "",
            type="password",
            help="Your OpenAI API key for embeddings and LLM"
        )
        
        pinecone_key = st.text_input(
            "Pinecone API Key", 
            value=Config.PINECONE_API_KEY or "",
            type="password",
            help="Your Pinecone API key for vector storage"
        )
        
        # Update environment if keys provided
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key
            Config.OPENAI_API_KEY = openai_key
        
        if pinecone_key:
            os.environ['PINECONE_API_KEY'] = pinecone_key
            Config.PINECONE_API_KEY = pinecone_key
        
        # System status
        st.subheader("📊 System Status")
        
        api_valid, api_message = validate_api_keys()
        if api_valid:
            st.success("✅ API Keys: Valid")
        else:
            st.error(f"❌ API Keys: {api_message}")
        
        if st.session_state.system_ready:
            st.success("✅ System: Ready")
        else:
            st.warning("⏳ System: Not Ready")
        
        # Documentation loading section
        st.subheader("📚 Documentation Management")
        
        max_pages = st.slider(
            "Max pages to crawl",
            min_value=5,
            max_value=100,
            value=Config.MAX_PAGES,
            help="Maximum number of documentation pages to crawl"
        )
        
        # Update config
        Config.MAX_PAGES = max_pages
        
        if st.button("🔄 Load ChaiCode Documentation", disabled=not api_valid):
            load_documentation()
        
        # System statistics
        if st.session_state.stats['total_queries'] > 0:
            st.subheader("📈 Usage Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", st.session_state.stats['total_queries'])
                st.metric("Documents", st.session_state.stats['documents_processed'])
            
            with col2:
                success_rate = (st.session_state.stats['successful_queries'] / 
                              st.session_state.stats['total_queries'] * 100)
                st.metric("Success Rate", f"{success_rate:.1f}%")
                st.metric("Chunks", st.session_state.stats['chunks_created'])
        
        # Advanced options (collapsible)
        with st.expander("🔧 Advanced Options"):
            
            st.subheader("Chunking Parameters")
            chunk_size = st.slider("Chunk Size", 500, 2000, Config.CHUNK_SIZE)
            chunk_overlap = st.slider("Chunk Overlap", 50, 500, Config.CHUNK_OVERLAP)
            
            st.subheader("Retrieval Parameters")
            retrieval_k = st.slider("Number of chunks to retrieve", 1, 10, Config.RETRIEVAL_K)
            
            # Update config
            Config.CHUNK_SIZE = chunk_size
            Config.CHUNK_OVERLAP = chunk_overlap
            Config.RETRIEVAL_K = retrieval_k
            
            if st.button("🔄 Reset System"):
                reset_system()

def load_documentation():
    """Load and process ChaiCode documentation"""
    
    progress_container = st.container()
    
    with progress_container:
        st.header("📖 Loading Documentation...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Initialize crawler
            status_text.text("🕷️ Initializing web crawler...")
            crawler = ChaiCodeDocsCrawler()
            progress_bar.progress(10)
            
            # Step 2: Crawl documentation
            status_text.text(f"📄 Crawling documentation (max {Config.MAX_PAGES} pages)...")
            doc_pages = crawler.crawl_docs(max_pages=Config.MAX_PAGES)
            progress_bar.progress(30)
            
            if not doc_pages:
                st.error("❌ No documentation pages found!")
                return
            
            # Step 3: Convert to LangChain documents
            status_text.text("📝 Converting documents...")
            documents = convert_to_langchain_docs(doc_pages)
            progress_bar.progress(40)
            
            # Step 4: Initialize embedding manager
            status_text.text("🧠 Setting up embeddings...")
            embedding_manager = EmbeddingManager()
            st.session_state.embedding_manager = embedding_manager
            progress_bar.progress(50)
            
            # Step 5: Chunk documents
            status_text.text("✂️ Chunking documents...")
            chunker = DocumentChunker()
            chunks = chunker.chunk_documents(documents)
            progress_bar.progress(60)
            
            # Step 6: Optimize chunks
            status_text.text("🔧 Optimizing chunks for retrieval...")
            optimized_chunks = optimize_chunks_for_retrieval(chunks)
            progress_bar.progress(70)
            
            # Step 7: Create vector store
            status_text.text("🗃️ Creating vector database...")
            vectorstore = embedding_manager.create_vectorstore(optimized_chunks)
            st.session_state.vectorstore = vectorstore
            progress_bar.progress(80)
            
            # Step 8: Setup query processor
            status_text.text("🤖 Initializing query processor...")
            query_processor = ChaiCodeQueryProcessor(embedding_manager)
            query_processor.setup_qa_chain(vectorstore)
            st.session_state.query_processor = query_processor
            progress_bar.progress(90)
            
            # Step 9: Finalize
            status_text.text("✅ Finalizing setup...")
            st.session_state.system_ready = True
            st.session_state.docs_loaded = True
            
            # Update statistics
            st.session_state.stats.update({
                'documents_processed': len(documents),
                'chunks_created': len(optimized_chunks)
            })
            
            progress_bar.progress(100)
            status_text.text("🎉 Documentation loaded successfully!")
            
            # Show success metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Documents", len(documents))
            with col2:
                st.metric("📝 Chunks", len(optimized_chunks))
            with col3:
                st.metric("🔗 Sources", len(set(doc['url'] for doc in doc_pages)))
            
            st.success("✅ System is now ready! You can start asking questions.")
            
            # Auto-refresh to update the main interface
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error loading documentation: {str(e)}")
            st.exception(e)

def reset_system():
    """Reset the entire system"""
    
    # Clear session state
    for key in ['embedding_manager', 'query_processor', 'vectorstore', 
                'system_ready', 'docs_loaded', 'chat_history']:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reset statistics
    st.session_state.stats = {
        'total_queries': 0,
        'successful_queries': 0,
        'documents_processed': 0,
        'chunks_created': 0
    }
    
    st.success("🔄 System reset successfully!")
    st.rerun()

def display_chat_interface():
    """Display the main chat interface"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 ChaiCode Documentation Assistant</h1>
        <p>Get instant answers from ChaiCode documentation with source references</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.system_ready:
        display_welcome_screen()
        return
    
    # Chat interface
    st.subheader("💬 Ask a Question")
    
    # Query input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_question = st.text_input(
            "Your question:",
            placeholder="e.g., How do I get started with ChaiCode?",
            key="user_input"
        )
    
    with col2:
        ask_button = st.button("🔍 Ask", type="primary")
    
    # Process query
    if ask_button and user_question:
        process_user_query(user_question)
    
    # Suggested questions
    display_suggested_questions()
    
    # Chat history
    display_chat_history()

def display_welcome_screen():
    """Display welcome screen when system is not ready"""
    
    prompts = create_system_prompts()
    
    st.markdown(prompts['welcome'])
    
    # Setup instructions
    st.subheader("🚀 Quick Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Step 1: Get API Keys**
        - [OpenAI API Key](https://platform.openai.com/api-keys)
        - [Pinecone API Key](https://www.pinecone.io/)
        """)
    
    with col2:
        st.markdown("""
        **Step 2: Load Documentation**
        - Enter API keys in sidebar
        - Click "Load ChaiCode Documentation"
        - Wait for processing to complete
        """)
    
    # Show system requirements
    with st.expander("📋 System Requirements"):
        st.markdown("""
        - **OpenAI API**: For embeddings and language model
        - **Pinecone**: For vector database storage
        - **Internet**: To crawl ChaiCode documentation
        - **Resources**: Processing may take 2-5 minutes depending on document count
        """)

def display_suggested_questions():
    """Display suggested questions for users"""
    
    if not st.session_state.query_processor:
        return
    
    st.subheader("💡 Suggested Questions")
    
    suggested = st.session_state.query_processor.get_suggested_questions()
    
    # Display in columns for better layout
    cols = st.columns(2)
    
    for i, question in enumerate(suggested[:8]):  # Show first 8 questions
        col = cols[i % 2]
        
        with col:
            if st.button(question, key=f"suggested_{i}"):
                process_user_query(question)

def process_user_query(question: str):
    """Process user query and display response"""
    
    if not st.session_state.query_processor:
        st.error("❌ System not ready. Please load documentation first.")
        return
    
    # Update statistics
    st.session_state.stats['total_queries'] += 1
    
    with st.spinner("🔍 Searching documentation..."):
        
        try:
            # Process query
            result = st.session_state.query_processor.query(question)
            
            # Add to chat history
            st.session_state.chat_history.append({
                'question': question,
                'result': result,
                'timestamp': time.time()
            })
            
            # Update success statistics
            if result.get('confidence', 0) > 0.5:
                st.session_state.stats['successful_queries'] += 1
            
            # Display result
            display_query_result(question, result)
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.exception(e)

def display_query_result(question: str, result: Dict[str, Any]):
    """Display the query result with formatting"""
    
    # Main answer
    st.subheader("📝 Answer")
    
    # Confidence indicator
    confidence = result.get('confidence', 0.5)
    if confidence > 0.8:
        confidence_color = "🟢"
        confidence_text = "High"
    elif confidence > 0.5:
        confidence_color = "🟡"
        confidence_text = "Medium"
    else:
        confidence_color = "🔴"
        confidence_text = "Low"
    
    st.markdown(f"**Confidence:** {confidence_color} {confidence_text} ({confidence:.1f})")
    
    # Answer text
    st.markdown(f"""
    <div class="chat-message">
        {result['answer']}
    </div>
    """, unsafe_allow_html=True)
    
    # Sources section
    sources = result.get('sources', [])
    if sources:
        st.subheader(f"📚 Sources ({len(sources)} found)")
        
        for i, source in enumerate(sources, 1):
            with st.expander(f"📄 Source {i}: {source['title']}", expanded=i==1):
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**URL:** [{source['url']}]({source['url']})")
                    st.markdown(f"**Preview:** {source['content_preview']}")
                
                with col2:
                    if 'relevance_score' in source:
                        st.metric("Relevance", f"{source['relevance_score']:.2f}")
                    
                    if source.get('keywords'):
                        st.markdown("**Keywords:**")
                        st.markdown(", ".join(source['keywords'][:5]))
    
    # Query metadata
    with st.expander("🔍 Query Details"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Query Type", result.get('query_type', 'general'))
        
        with col2:
            st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
        
        with col3:
            st.metric("Chunks Retrieved", result.get('retrieved_chunks', 0))

def display_chat_history():
    """Display chat history"""
    
    if not st.session_state.chat_history:
        return
    
    st.subheader("💭 Recent Questions")
    
    # Show last 5 queries
    recent_queries = st.session_state.chat_history[-5:]
    
    for i, chat in enumerate(reversed(recent_queries)):
        with st.expander(f"Q: {chat['question'][:60]}..."):
            st.markdown(f"**Answer:** {chat['result']['answer'][:200]}...")
            
            if chat['result'].get('sources'):
                st.markdown(f"**Sources:** {len(chat['result']['sources'])} found")
            
            # Replay button
            if st.button(f"🔄 Ask Again", key=f"replay_{i}"):
                process_user_query(chat['question'])

def main():
    """Main entry point for the ChaiCode Docs Assistant Streamlit app"""
    
    # Initialize session state variables
    initialize_session_state()
    
    # Render sidebar with configuration and controls
    setup_sidebar()
    
    # Render main content based on system readiness
    if not st.session_state.system_ready:
        display_welcome_screen()
    else:
        display_chat_interface()

# Run the app
if __name__ == "__main__":
    main()

               