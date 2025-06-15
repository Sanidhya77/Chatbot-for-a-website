import os
import time
import logging
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
load_dotenv()
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the chatbot application"""

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'chaicode-docs')
    PINECONE_NAMESPACE = os.getenv('PINECONE_NAMESPACE', 'default')
    BASE_URL = os.getenv('BASE_URL', 'https://docs.chaicode.com/youtube/getting-started/')
    START_URL = os.getenv('START_URL', 'https://docs.chaicode.com/')

    LLM_TEMPERATURE = 0.1
    MAX_TOKENS = 4000
    RETRIEVAL_K = 5

    MAX_PAGES = 50
    CRAWL_DELAY = 1.0
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    SIDEBAR_WIDTH = 300
    MAX_CHAT_HISTORY = 100

    def __init__(self):
        self.pinecone_api_key = self.PINECONE_API_KEY
        self.pinecone_index_name = self.PINECONE_INDEX_NAME
        self.pinecone_namespace = self.PINECONE_NAMESPACE
        self.openai_api_key = self.OPENAI_API_KEY

        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

    def get_pinecone_config(self):
        return {
            'api_key': self.pinecone_api_key,
            'index_name': self.pinecone_index_name,
            'namespace': self.pinecone_namespace
        }

    def get_openai_config(self):
        return {
            'api_key': self.openai_api_key,
            'max_tokens': self.MAX_TOKENS,
            'temperature': self.LLM_TEMPERATURE
        }


class EmbeddingManager:
    """Manages embeddings and vector store operations"""

    def __init__(self, config=None):
        self.config = config or Config()

        self.pinecone_api_key = self.config.pinecone_api_key
        self.pinecone_index_name = self.config.pinecone_index_name
        self.pinecone_namespace = self.config.pinecone_namespace
        self.openai_api_key = self.config.openai_api_key

        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_api_key
        )

        # Dynamically get embedding dimension
        self.embedding_dimension = len(self.embedding_model.embed_query("hello"))
        self.pinecone_index = None
        self.vectorstore = None

    def _to_documents(self, raw_docs):
        return [
            Document(
                page_content=doc.get('content', ''),
                metadata=doc.get('metadata', {})
            ) for doc in raw_docs
        ]

    def initialize_index(self):
        try:
            existing_indexes = self.pc.list_indexes().names()
            if self.pinecone_index_name in existing_indexes:
                self.pinecone_index = self.pc.Index(self.pinecone_index_name)
                logger.info(f"Connected to existing index: {self.pinecone_index_name}")
            else:
                logger.info(f"Creating new index: {self.pinecone_index_name}")
                self.pc.create_index(
                    name=self.pinecone_index_name,
                    dimension=self.embedding_dimension,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )

                logger.info("Waiting for index to be ready...")
                while True:
                    status = self.pc.describe_index(self.pinecone_index_name).status
                    if status['ready']:
                        break
                    time.sleep(2)

                self.pinecone_index = self.pc.Index(self.pinecone_index_name)
                logger.info("Index created and ready!")

        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {e}")
            raise

    def create_vectorstore(self, documents):
        self.vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=self.embedding_model,
    index_name=self.pinecone_index_name,  # Use name instead of object
    namespace=self.pinecone_namespace
)


    def get_vectorstore(self):
        self.vectorstore = PineconeVectorStore(
    index_name=self.pinecone_index_name,
    embedding=self.embedding_model,
    namespace=self.pinecone_namespace
)


    def similarity_search(self, query, k=None):
        if self.vectorstore is None:
            self.get_vectorstore()

        try:
            return self.vectorstore.similarity_search(query, k=k or self.config.RETRIEVAL_K)
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def add_documents(self, documents):
        if not documents:
            logger.warning("No documents to add.")
            return

        if all(isinstance(doc, dict) for doc in documents):
            documents = self._to_documents(documents)

        try:
            self.get_vectorstore().add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vectorstore.")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def delete_index(self):
        try:
            self.pc.delete_index(self.pinecone_index_name)
            logger.warning(f"Deleted index: {self.pinecone_index_name}")
            self.pinecone_index = None
            self.vectorstore = None
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            raise

# Only load environment if running this file directly
if __name__ == "__main__":
    load_dotenv()
    config = Config()
    manager = EmbeddingManager(config)
    manager.initialize_index()
