# Chatbot-for-a-website
🧠 ChaiCode Docs Chatbot
An AI-powered chatbot that allows users to interactively query the official ChaiCode documentation using natural language. Built with LangChain, OpenAI, Pinecone, and Streamlit, this tool enables semantic search over documentation with Retrieval-Augmented Generation (RAG).

🚀 Features
📄 PDF/Web Docs Ingestion — Easily load ChaiCode docs via crawling or upload

🧠 Semantic Search — Vector-based retrieval using OpenAI embeddings

💬 Natural Language Chat Interface — Ask questions conversationally via Streamlit UI

🧲 Vector Store with Pinecone — Efficient document chunk storage and retrieval

📦 Modular Design — Easy to extend with new embeddings, LLMs, or vector DBs

🛠️ Tech Stack
LangChain for chaining document loaders, retrievers, and LLMs

OpenAI Embeddings (text-embedding-3-small) for dense vector representation

Pinecone for vector database and similarity search

Streamlit for interactive chat frontend

BeautifulSoup & Requests for crawling docs

Python for backend glue and modular logic

⚙️ Setup
bash
Copy
Edit
git clone https://github.com/yourusername/chaicode-docs-chatbot.git
cd chaicode-docs-chatbot
pip install -r requirements.txt
Create a .env file:

env
Copy
Edit
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=chaicode-docs
PINECONE_NAMESPACE=default
BASE_URL=https://docs.chaicode.com/youtube/getting-started/
START_URL=https://docs.chaicode.com/
Run the app:

bash
Copy
Edit
streamlit run app.py
📚 Example Use Case
Ask:

"How do I set up a YouTube clone with ChaiCode?"
And the bot retrieves relevant sections of the documentation and gives an LLM-powered answer.

