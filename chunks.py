# chunks.py
"""Document chunking and text processing for ChaiCode RAG system"""

from typing import List, Dict, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import Config
import re

class DocumentChunker:
    """Handles document chunking with various strategies"""
    
    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
        
        # Initialize different text splitters
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.markdown_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]
        )
    
    def preprocess_document(self, doc: Document) -> Document:
        """Preprocess document content before chunking"""
        content = doc.page_content
        
        # Clean up content
        content = self._clean_content(content)
        
        # Add structure markers for better chunking
        content = self._add_structure_markers(content, doc.metadata)
        
        # Create new document with cleaned content
        return Document(
            page_content=content,
            metadata=doc.metadata.copy()
        )
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Fix common formatting issues
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Multiple newlines
        content = re.sub(r'(\w)([A-Z])', r'\1 \2', content)  # Missing spaces before capitals
        
        # Remove page navigation elements that might have slipped through
        navigation_patterns = [
            r'Previous\s*Next',
            r'Home\s*>\s*.*',
            r'Skip to main content',
            r'Table of contents',
            r'Edit this page'
        ]
        
        for pattern in navigation_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _add_structure_markers(self, content: str, metadata: Dict) -> str:
        """Add context markers to help with chunking"""
        # Add document title at the beginning
        title = metadata.get('title', 'Unknown')
        url = metadata.get('url', '')
        
        # Create header with context
        header = f"Document: {title}\nSource: {url}\n\n"
        
        return header + content
    
    def chunk_documents(self, documents: List[Document], strategy: str = "recursive") -> List[Document]:
        """Chunk documents using specified strategy"""
        all_chunks = []
        
        print(f"🔪 Starting document chunking with strategy: {strategy}")
        print(f"📏 Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        for i, doc in enumerate(documents):
            # Preprocess document
            processed_doc = self.preprocess_document(doc)
            
            # Choose splitter based on strategy
            if strategy == "markdown":
                chunks = self.markdown_splitter.split_documents([processed_doc])
            else:  # default to recursive
                chunks = self.recursive_splitter.split_documents([processed_doc])
            
            # Add chunk metadata
            for j, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': f"{i}_{j}",
                    'chunk_index': j,
                    'total_chunks': len(chunks),
                    'source_doc_index': i,
                    'chunk_strategy': strategy
                })
            
            all_chunks.extend(chunks)
            
            if (i + 1) % 10 == 0:
                print(f"📄 Processed {i + 1}/{len(documents)} documents")
        
        print(f"✅ Chunking completed!")
        print(f"📊 Original documents: {len(documents)}")
        print(f"📊 Total chunks created: {len(all_chunks)}")
        print(f"📊 Average chunks per document: {len(all_chunks) / len(documents):.1f}")
        
        return all_chunks
    
    def analyze_chunks(self, chunks: List[Document]) -> Dict:
        """Analyze chunk statistics"""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        word_counts = [len(chunk.page_content.split()) for chunk in chunks]
        
        # Get unique sources
        sources = set(chunk.metadata.get('url', 'unknown') for chunk in chunks)
        
        analysis = {
            'total_chunks': len(chunks),
            'unique_sources': len(sources),
            'chunk_length_stats': {
                'min': min(chunk_lengths),
                'max': max(chunk_lengths),
                'avg': sum(chunk_lengths) / len(chunk_lengths),
                'median': sorted(chunk_lengths)[len(chunk_lengths) // 2]
            },
            'word_count_stats': {
                'min': min(word_counts),
                'max': max(word_counts),
                'avg': sum(word_counts) / len(word_counts),
                'median': sorted(word_counts)[len(word_counts) // 2]
            }
        }
        
        return analysis
    
    def filter_chunks(self, chunks: List[Document], min_length: int = 100) -> List[Document]:
        """Filter out chunks that are too short or low quality"""
        filtered_chunks = []
        
        for chunk in chunks:
            content = chunk.page_content.strip()
            
            # Skip very short chunks
            if len(content) < min_length:
                continue
            
            # Skip chunks that are mostly navigation or boilerplate
            if self._is_low_quality_chunk(content):
                continue
            
            filtered_chunks.append(chunk)
        
        print(f"🔍 Filtered chunks: {len(chunks)} → {len(filtered_chunks)}")
        return filtered_chunks
    
    def _is_low_quality_chunk(self, content: str) -> bool:
        """Determine if a chunk is low quality (navigation, boilerplate, etc.)"""
        content_lower = content.lower()
        
        # Common patterns for low-quality chunks
        low_quality_patterns = [
            'click here',
            'read more',
            'see also',
            'table of contents',
            'navigation',
            'breadcrumb',
            'skip to',
            'go to',
            'page not found',
            '404 error',
            'coming soon'
        ]
        
        # Check if chunk is mostly low-quality patterns
        pattern_count = sum(1 for pattern in low_quality_patterns if pattern in content_lower)
        
        # If more than 20% of the content matches low-quality patterns
        if pattern_count > len(low_quality_patterns) * 0.2:
            return True
        
        # Check if chunk has very few actual words
        words = content.split()
        if len(words) < 10:
            return True
        
        # Check if chunk is mostly repeated characters or symbols
        unique_chars = len(set(content.replace(' ', '').replace('\n', '')))
        if unique_chars < 10:
            return True
        
        return False
    
    def create_chunk_summaries(self, chunks: List[Document]) -> List[Document]:
        """Create summaries for chunks to improve retrieval"""
        # This could be enhanced with an LLM to create actual summaries
        # For now, we'll create simple keyword-based summaries
        
        for chunk in chunks:
            # Extract key information for summary
            content = chunk.page_content
            title = chunk.metadata.get('title', '')
            
            # Simple keyword extraction (could be improved with NLP)
            words = content.lower().split()
            word_freq = {}
            
            for word in words:
                if len(word) > 3 and word.isalpha():  # Filter short words and non-alphabetic
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            keywords = [word for word, freq in top_keywords]
            
            # Add summary to metadata
            chunk.metadata['keywords'] = keywords
            chunk.metadata['summary'] = f"Content from {title}: {', '.join(keywords[:5])}"
        
        return chunks

def optimize_chunks_for_retrieval(chunks: List[Document]) -> List[Document]:
    """Optimize chunks specifically for better retrieval performance"""
    chunker = DocumentChunker()
    
    # Filter low-quality chunks
    filtered_chunks = chunker.filter_chunks(chunks)
    
    # Add summaries and keywords
    optimized_chunks = chunker.create_chunk_summaries(filtered_chunks)
    
    return optimized_chunks

# Test function
def test_chunking():
    """Test the chunking functionality"""
    from langchain.schema import Document
    
    # Create test documents
    test_docs = [
        Document(
            page_content="This is a test document about ChaiCode. It contains information about getting started with the platform. ChaiCode is a comprehensive learning platform for developers. It offers various courses and tutorials.",
            metadata={'title': 'Getting Started', 'url': 'https://example.com/start'}
        ),
        Document(
            page_content="Advanced ChaiCode features include project management, collaboration tools, and integrated development environments. These features help developers work more efficiently and effectively on their projects.",
            metadata={'title': 'Advanced Features', 'url': 'https://example.com/advanced'}
        )
    ]
    
    chunker = DocumentChunker()
    
    # Test chunking
    chunks = chunker.chunk_documents(test_docs)
    
    if chunks:
        print(f"✅ Chunking test successful! Created {len(chunks)} chunks")
        
        # Analyze chunks
        analysis = chunker.analyze_chunks(chunks)
        print(f"📊 Chunk analysis: {analysis}")
        
        # Show sample chunks
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"Content: {chunk.page_content[:100]}...")
            print(f"Metadata: {chunk.metadata}")
        
        return True
    else:
        print("❌ Chunking test failed!")
        return False

if __name__ == "__main__":
    test_chunking()

#PubSub / Queue / Streaming / Batch Process / Cron Jobs
