# query.py
"""Query processing and response generation for ChaiCode RAG system"""

from typing import Dict, List, Any
from openai import OpenAI as OpenAIClient
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config, EmbeddingManager
import time
import json

class ChaiCodeQueryProcessor:
    """Handles query processing and response generation"""
    
    def __init__(self, embedding_manager: EmbeddingManager, config: Config = None):
        self.embedding_manager = embedding_manager
        self.config = config or Config()
        self.openai_client = None
        self.vectorstore = None
        self.setup_openai_client()
        self.setup_prompts()
    
    def setup_openai_client(self):
        """Initialize the OpenAI client"""
        import os
        
        # Get API key from config instance
        api_key = self.config.openai_api_key
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables."
            )
        
        self.openai_client = OpenAIClient(
            api_key=api_key
        )
        print("✅ OpenAI client initialized")
    
    def setup_prompts(self):
        """Setup custom prompts for different query types"""
        
        # Main QA prompt
        self.qa_prompt_template = """You are a helpful assistant for ChaiCode documentation. 
Use the following pieces of context to answer the question at the end. 

Guidelines:
1. Provide accurate, helpful answers based on the context
2. If you don't know the answer from the context, say so
3. Include specific details and examples when available
4. Be concise but comprehensive
5. Format your response clearly with proper structure
6. Always mention the source URL when providing information

Context:
{context}

Question: {question}

Answer (include source URL):"""

        # Getting started prompt
        self.getting_started_prompt_template = """You are a ChaiCode expert helping new users get started.
Based on the provided documentation context, give a step-by-step guide for beginners.

Context:
{context}

Question: {question}

Provide a beginner-friendly answer with:
1. Clear step-by-step instructions
2. Prerequisites if any
3. Common pitfalls to avoid
4. Next steps or recommendations
5. Relevant documentation links

Answer:"""

        # Troubleshooting prompt
        self.troubleshooting_prompt_template = """You are a ChaiCode technical support expert.
Help users solve problems based on the documentation context.

Context:
{context}

Problem: {question}

Provide a troubleshooting response with:
1. Possible causes of the issue
2. Step-by-step solutions
3. Alternative approaches if main solution doesn't work
4. Prevention tips
5. Related documentation links

Solution:"""
    
    def setup_qa_chain(self, vectorstore):
        """Setup the vectorstore for retrieval"""
        self.vectorstore = vectorstore
        print("✅ Vectorstore setup completed")
    
    def classify_query(self, question: str) -> str:
        """Classify the type of query to use appropriate prompt"""
        question_lower = question.lower()
        
        # Getting started queries
        getting_started_keywords = [
            'get started', 'getting started', 'begin', 'start', 
            'how to start', 'first time', 'beginner', 'new user',
            'setup', 'installation', 'install'
        ]
        
        # Troubleshooting queries
        troubleshooting_keywords = [
            'error', 'problem', 'issue', 'not working', 'broken',
            'fix', 'solve', 'troubleshoot', 'debug', 'help',
            'failed', 'can\'t', 'cannot', 'won\'t', 'doesn\'t work'
        ]
        
        if any(keyword in question_lower for keyword in getting_started_keywords):
            return 'getting_started'
        elif any(keyword in question_lower for keyword in troubleshooting_keywords):
            return 'troubleshooting'
        else:
            return 'general'
    
    def enhance_query(self, question: str) -> str:
        """Enhance query for better retrieval"""
        # Add context keywords that might help retrieval
        enhanced_question = question
        
        # Add ChaiCode context if not present
        if 'chaicode' not in question.lower():
            enhanced_question = f"ChaiCode {question}"
        
        return enhanced_question
    
    def retrieve_documents(self, question: str, k: int = None) -> List[Document]:
        """Retrieve relevant documents from vectorstore"""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call setup_qa_chain first.")
        
        k = k or 5  # Default value since we're not using Config.RETRIEVAL_K
        
        try:
            # Use similarity search with score threshold
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                question, k=k
            )
            
            # Filter by score threshold and return documents
            relevant_docs = []
            for doc, score in docs_with_scores:
                if score >= 0.7:  # Only return chunks with good similarity
                    # Add relevance score to metadata
                    doc.metadata['relevance_score'] = score
                    relevant_docs.append(doc)
            
            return relevant_docs
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def generate_response(self, question: str, context_docs: List[Document], query_type: str) -> str:
        """Generate response using OpenAI API"""
        try:
            # Prepare context from documents
            context = "\n\n".join([
                f"Source: {doc.metadata.get('url', 'https://docs.chaicode.com/youtube/getting-started/')}\n"
                f"Title: {doc.metadata.get('title', 'Chaicode docs ')}\n"
                f"Content: {doc.page_content}"
                for doc in context_docs
            ])
            
            # Choose appropriate prompt based on query type
            if query_type == 'getting_started':
                prompt_template = self.getting_started_prompt_template
            elif query_type == 'troubleshooting':
                prompt_template = self.troubleshooting_prompt_template
            else:
                prompt_template = self.qa_prompt_template
            
            # Format the prompt
            formatted_prompt = prompt_template.format(
                context=context,
                question=question
            )
            
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful documentation assistant."},
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I apologize, but I encountered an error while generating the response: {str(e)}"
    
    def query(self, question: str, query_type: str = None) -> Dict[str, Any]:
        """Process a query and return response with sources"""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Call setup_qa_chain first.")
        
        start_time = time.time()
        
        try:
            # Classify query type if not provided
            if not query_type:
                query_type = self.classify_query(question)
            
            # Enhance query for better retrieval
            enhanced_question = self.enhance_query(question)
            
            # Retrieve relevant documents
            source_docs = self.retrieve_documents(enhanced_question)
            
            if not source_docs:
                return {
                    'answer': "I couldn't find relevant information in the ChaiCode documentation to answer your question. Please try rephrasing your question or check the main ChaiCode website for additional resources.",
                    'sources': [],
                    'query_type': query_type,
                    'confidence': 0.0,
                    'processing_time': time.time() - start_time,
                    'original_question': question,
                    'total_sources': 0,
                    'retrieved_chunks': 0
                }
            
            # Generate response
            answer = self.generate_response(question, source_docs, query_type)
            
            # Process response
            response = self._process_response(answer, source_docs, question, query_type)
            response['processing_time'] = time.time() - start_time
            
            return response
            
        except Exception as e:
            return {
                'answer': f"I apologize, but I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'query_type': query_type,
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'error': str(e),
                'original_question': question,
                'total_sources': 0,
                'retrieved_chunks': 0
            }
    
    def _process_response(self, answer: str, source_docs: List[Document], original_question: str, query_type: str) -> Dict[str, Any]:
        """Process and format the response"""
        
        # Extract and format sources
        sources = []
        seen_urls = set("https://docs.chaicode.com/youtube/getting-started/")
        
        for doc in source_docs:
            url = doc.metadata.get('url', 'https://docs.chaicode.com/youtube/getting-started/')
            title = doc.metadata.get('title', 'ChaiCode Documentation')
            
            if url and url not in seen_urls:
                source_info = {
                    'url': url,
                    'title': title,
                    'content_preview': doc.page_content[:200] + "...",
                    'relevance_score': doc.metadata.get('relevance_score', 0.8),
                    'chunk_id': doc.metadata.get('chunk_id', ''),
                    'keywords': doc.metadata.get('keywords', [])
                }
                sources.append(source_info)
                seen_urls.add(url)
        
        # Calculate confidence based on source quality and answer length
        confidence = self._calculate_confidence(answer, sources, source_docs)
        
        return {
            'answer': answer,
            'sources': sources,
            'query_type': query_type,
            'confidence': confidence,
            'original_question': original_question,
            'total_sources': len(sources),
            'retrieved_chunks': len(source_docs)
        }
    
    def _calculate_confidence(self, answer: str, sources: List[Dict], source_docs: List) -> float:
        """Calculate confidence score for the response"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on number of sources
        if len(sources) >= 3:
            confidence += 0.2
        elif len(sources) >= 2:
            confidence += 0.1
        
        # Boost confidence based on answer length (more detailed = higher confidence)
        if len(answer) > 200:
            confidence += 0.1
        elif len(answer) > 100:
            confidence += 0.05
        
        # Reduce confidence if answer is generic
        generic_phrases = [
            "I don't know", "I'm not sure", "I couldn't find", 
            "I apologize", "not available", "no information"
        ]
        
        if any(phrase in answer.lower() for phrase in generic_phrases):
            confidence -= 0.3
        
        # Boost confidence if answer includes specific details
        if any(keyword in answer.lower() for keyword in ['step', 'example', 'code', 'command']):
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
    
    def get_suggested_questions(self) -> List[str]:
        """Return a list of suggested questions users might ask"""
        return [
            "How do I get started with ChaiCode?",
            "What are the main features of ChaiCode?",
            "How do I set up my development environment?",
            "What programming languages does ChaiCode support?",
            "How do I create my first project in ChaiCode?",
            "What are the system requirements for ChaiCode?",
            "How do I access ChaiCode tutorials?",
            "Can I collaborate with others on ChaiCode?",
            "How do I troubleshoot common ChaiCode issues?",
            "Where can I find ChaiCode documentation?",
            "How do I update ChaiCode to the latest version?",
            "What are the best practices for using ChaiCode?"
        ]
    
    def search_similar_questions(self, question: str, threshold: float = 0.8) -> List[Dict]:
        """Find similar questions that have been asked before"""
        if not self.vectorstore:
            return []
        
        try:
            # Search for similar content
            similar_docs = self.vectorstore.similarity_search_with_score(
                question, k=5
            )
            
            similar_questions = []
            for doc, score in similar_docs:
                if score >= threshold:
                    similar_questions.append({
                        'content': doc.page_content[:200] + "...",
                        'title': doc.metadata.get('title', ''),
                        'url': doc.metadata.get('url', ''),
                        'similarity_score': score
                    })
            
            return similar_questions
            
        except Exception as e:
            print(f"Error finding similar questions: {e}")
            return []

def create_system_prompts() -> Dict[str, str]:
    """Create system prompts for different scenarios"""
    
    prompts = {
        'welcome': """
        Welcome to the ChaiCode Documentation Assistant! 🤖
        
        I'm here to help you navigate and understand ChaiCode documentation quickly and efficiently.
        
        I can help you with:
        • Getting started guides and tutorials
        • Feature explanations and usage
        • Troubleshooting common issues
        • Best practices and recommendations
        • Finding specific documentation sections
        
        Just ask me anything about ChaiCode, and I'll provide detailed answers with direct links to the relevant documentation!
        """,
        
        'no_results': """
        I couldn't find specific information about your question in the current ChaiCode documentation.
        
        This might happen because:
        • The topic might be covered in a different section
        • The documentation might not include this specific information yet
        • Your question might need to be rephrased
        
        You can try:
        • Rephrasing your question with different keywords
        • Breaking down complex questions into simpler parts
        • Checking the main ChaiCode website for additional resources
        
        Would you like to try asking your question differently?
        """,
        
        'error_handling': """
        I apologize, but I encountered an issue while processing your question.
        
        This might be due to:
        • Temporary connectivity issues
        • High system load
        • An unexpected error in processing
        
        Please try:
        • Asking your question again
        • Using simpler language or fewer technical terms
        • Breaking complex questions into smaller parts
        
        If the problem persists, you can access the ChaiCode documentation directly at docs.chaicode.com
        """
    }
    
    return prompts

# Test function
def test_query_processor():
    """Test the query processor functionality"""
    try:
        # This would normally require a properly set up embedding manager
        # For testing, we'll just test the classification and enhancement functions
        
        # Mock embedding manager for testing
        class MockEmbeddingManager:
            pass
        
        processor = ChaiCodeQueryProcessor(MockEmbeddingManager())
        
        # Test query classification
        test_questions = [
            "How do I get started with ChaiCode?",
            "I'm having an error with my installation",
            "What are the features of ChaiCode?"
        ]
        
        for question in test_questions:
            query_type = processor.classify_query(question)
            enhanced = processor.enhance_query(question)
            print(f"Question: {question}")
            print(f"Type: {query_type}")
            print(f"Enhanced: {enhanced}\n")
        
        # Test suggested questions
        suggestions = processor.get_suggested_questions()
        print(f"✅ Generated {len(suggestions)} suggested questions")
        
        return True
        
    except Exception as e:
        print(f"❌ Query processor test failed: {e}")
        return False

if __name__ == "__main__":
    test_query_processor()