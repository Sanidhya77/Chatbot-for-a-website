# ingest_docs.py
"""Document ingestion and web crawling for ChaiCode documentation"""

import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Set
import re
from langchain.schema import Document

class Config:
    """Configuration class with missing values added"""
    BASE_URL = 'https://docs.chaicode.com/youtube/getting-started/'
    START_URL = 'https://docs.chaicode.com/'
    MAX_PAGES = 50
    CRAWL_DELAY = 1.0
    MIN_CONTENT_LENGTH = 50  # Added missing config
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3

class ChaiCodeDocsCrawler:
    """Crawls and extracts content from ChaiCode documentation"""
    
    def __init__(self):
        self.base_url = Config.BASE_URL
        self.visited_urls: Set[str] = set()
        self.doc_pages: List[Dict] = []
        self.session = requests.Session()
        
        # Set headers to avoid blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def is_valid_doc_url(self, url: str) -> bool:
        """Check if URL is a valid documentation page"""
        if not url:
            return False
            
        parsed = urlparse(url)
        
        # Must be from docs.chaicode.com
        if parsed.netloc != "docs.chaicode.com":
            return False
        
        # Avoid non-content files
        excluded_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.ico', '.zip', '.tar', '.gz'}
        if any(url.lower().endswith(ext) for ext in excluded_extensions):
            return False
        
        # Avoid fragments and query parameters for deduplication
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
        return clean_url not in self.visited_urls
    
    def extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Extract all valid documentation links from a page"""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Skip empty hrefs, anchors, and mailto/tel links
            if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
                continue
                
            absolute_url = urljoin(current_url, href)
            
            if self.is_valid_doc_url(absolute_url):
                # Clean URL (remove fragments and query params)
                parsed = urlparse(absolute_url)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
                
                if clean_url not in self.visited_urls and clean_url not in links:
                    links.append(clean_url)
        
        return links
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove some problematic characters but keep most punctuation
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        return text.strip()
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract structured content from a documentation page"""
        # Remove unwanted elements
        unwanted_selectors = [
            'script', 'style', 'nav', 'footer', 'header', 
            '.navigation', '.sidebar', '.menu', '.breadcrumb',
            '.advertisement', '.ads', '.social-share', '.comments'
        ]
        
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Extract title with multiple fallback options
        title = "No Title"
        title_selectors = [
            'h1',
            '.page-title', 
            '.post-title',
            '.entry-title',
            'title',
            '.title'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem and title_elem.get_text().strip():
                title = self.clean_text(title_elem.get_text())
                break
        
        # If title is still generic, try to get it from the page title
        if title == "No Title":
            title_tag = soup.find('title')
            if title_tag:
                title = self.clean_text(title_tag.get_text())
        
        # Extract main content with multiple fallback selectors
        content = ""
        content_selectors = [
            'main',
            '.content',
            '.main-content',
            '.documentation', 
            '.markdown-body',
            'article',
            '.post-content',
            '.page-content',
            '.entry-content',
            '#content',
            '.wiki-content',
            '.document'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text(separator='\n', strip=True)
                if content.strip():  # Make sure we got actual content
                    break
        
        # Fallback to body if no specific content area found
        if not content.strip():
            body = soup.find('body')
            if body:
                content = body.get_text(separator='\n', strip=True)
        
        # Clean content
        content = self.clean_text(content)
        
        # Extract headings for better structure
        headings = []
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            heading_text = self.clean_text(heading.get_text())
            if heading_text and heading_text not in headings:
                headings.append(heading_text)
        
        return {
            'url': url,
            'title': title,
            'content': content,
            'headings': headings,
            'word_count': len(content.split()) if content else 0,
            'char_count': len(content) if content else 0
        }
    
    def crawl_docs(self, start_url: str = None, max_pages: int = None) -> List[Dict]:
        """Crawl documentation starting from a given URL"""
        start_url = start_url or Config.START_URL
        max_pages = max_pages or Config.MAX_PAGES
        
        urls_to_visit = [start_url]
        
        print(f"🕷️  Starting crawl from: {start_url}")
        print(f"📄 Max pages to crawl: {max_pages}")
        
        while urls_to_visit and len(self.doc_pages) < max_pages:
            current_url = urls_to_visit.pop(0)
            
            # Clean the URL for comparison
            parsed = urlparse(current_url)
            clean_current_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
            
            if clean_current_url in self.visited_urls:
                continue
            
            try:
                print(f"📖 Crawling ({len(self.doc_pages)+1}/{max_pages}): {current_url}")
                
                response = self.session.get(current_url, timeout=Config.REQUEST_TIMEOUT)
                response.raise_for_status()
                
                # Check if we got HTML content
                content_type = response.headers.get('content-type', '').lower()
                if 'html' not in content_type:
                    print(f"⏭️  Skipped (not HTML): {current_url}")
                    self.visited_urls.add(clean_current_url)
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract content
                doc_data = self.extract_content(soup, current_url)
                
                # Only include pages with substantial content
                if doc_data['word_count'] >= Config.MIN_CONTENT_LENGTH:
                    self.doc_pages.append(doc_data)
                    print(f"✅ Extracted: {doc_data['title'][:50]}... ({doc_data['word_count']} words)")
                else:
                    print(f"⏭️  Skipped (too short): {doc_data['title'][:50]}... ({doc_data['word_count']} words)")
                
                # Find new links to crawl
                new_links = self.extract_links(soup, current_url)
                
                # Add new links to the queue (limit to avoid infinite crawling)
                for link in new_links[:5]:  # Limit new links per page
                    if link not in urls_to_visit:
                        urls_to_visit.append(link)
                
                self.visited_urls.add(clean_current_url)
                
                # Be respectful with delays
                time.sleep(Config.CRAWL_DELAY)
                
            except requests.RequestException as e:
                print(f"❌ Request error for {current_url}: {str(e)}")
                self.visited_urls.add(clean_current_url)
                continue
            except Exception as e:
                print(f"❌ Unexpected error for {current_url}: {str(e)}")
                self.visited_urls.add(clean_current_url)
                continue
        
        print(f"\n🎉 Crawling completed!")
        print(f"📚 Total pages collected: {len(self.doc_pages)}")
        
        if self.doc_pages:
            total_words = sum(page['word_count'] for page in self.doc_pages)
            print(f"📊 Total words: {total_words}")
        else:
            print("❌ No documentation pages found!")
            print("🔍 Troubleshooting tips:")
            print("   - Check if the start URL is accessible")
            print("   - Verify the website structure hasn't changed")
            print("   - Try increasing MIN_CONTENT_LENGTH")
            print("   - Check your internet connection")
        
        return self.doc_pages
    
    def save_crawled_data(self, filename: str = "crawled_docs.json"):
        """Save crawled data to JSON file"""
        import json
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.doc_pages, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Crawled data saved to {filename}")
    
    def load_crawled_data(self, filename: str = "crawled_docs.json") -> List[Dict]:
        """Load previously crawled data from JSON file"""
        import json
        import os
        
        if not os.path.exists(filename):
            print(f"❌ File {filename} not found")
            return []
        
        with open(filename, 'r', encoding='utf-8') as f:
            self.doc_pages = json.load(f)
        
        print(f"📂 Loaded {len(self.doc_pages)} pages from {filename}")
        return self.doc_pages

def convert_to_langchain_docs(doc_pages: List[Dict]) -> List[Document]:
    """Convert crawled pages to LangChain Document objects"""
    documents = []
    
    for page in doc_pages:
        # Create comprehensive content
        full_content = f"Title: {page['title']}\n\n"
        
        if page.get('headings'):
            full_content += f"Headings: {', '.join(page['headings'])}\n\n"
        
        full_content += page['content']
        
        doc = Document(
            page_content=full_content,
            metadata={
                'url': page['url'],
                'title': page['title'],
                'word_count': page['word_count'],
                'char_count': page['char_count'],
                'headings': page.get('headings', [])
            }
        )
        documents.append(doc)
    
    print(f"📝 Converted {len(documents)} pages to LangChain documents")
    return documents

# Test function with better debugging
def test_crawler():
    """Test the crawler functionality with debugging"""
    print("🧪 Testing crawler...")
    
    crawler = ChaiCodeDocsCrawler()
    
    # First, test if we can reach the start URL
    try:
        print(f"🌐 Testing connection to: {Config.START_URL}")
        response = crawler.session.get(Config.START_URL, timeout=10)
        response.raise_for_status()
        print(f"✅ Connection successful! Status code: {response.status_code}")
        print(f"📄 Content type: {response.headers.get('content-type', 'unknown')}")
        print(f"📏 Content length: {len(response.content)} bytes")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # Test with a small number of pages
    test_pages = crawler.crawl_docs(max_pages=3)
    
    if test_pages:
        print(f"✅ Crawler test successful! Retrieved {len(test_pages)} pages")
        
        # Show sample data
        for i, page in enumerate(test_pages[:2]):
            print(f"\n📄 Page {i+1}:")
            print(f"   Title: {page['title']}")
            print(f"   URL: {page['url']}")
            print(f"   Word count: {page['word_count']}")
            print(f"   Content preview: {page['content'][:200]}...")
        
        return True
    else:
        print("❌ Crawler test failed!")
        print(f"🔍 Visited URLs: {list(crawler.visited_urls)}")
        return False

# Alternative test with specific URL
def test_specific_url(url: str = None):
    """Test crawling a specific URL"""
    test_url = url or "https://docs.chaicode.com/"
    
    print(f"🧪 Testing specific URL: {test_url}")
    
    crawler = ChaiCodeDocsCrawler()
    
    try:
        response = crawler.session.get(test_url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        doc_data = crawler.extract_content(soup, test_url)
        
        print(f"📄 Title: {doc_data['title']}")
        print(f"📊 Word count: {doc_data['word_count']}")
        print(f"📝 Content preview: {doc_data['content'][:300]}...")
        print(f"🔗 Found headings: {doc_data['headings'][:5]}")
        
        # Test link extraction
        links = crawler.extract_links(soup, test_url)
        print(f"🔗 Found {len(links)} links")
        
        return doc_data['word_count'] > 0
        
    except Exception as e:
        print(f"❌ Error testing URL: {e}")
        return False

if __name__ == "__main__":
    # Run both tests
    print("=" * 50)
    test_specific_url()
    print("=" * 50)
    test_crawler()