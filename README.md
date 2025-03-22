# ResearchChat AI
## Overview
ResearchChat AI is a Streamlit-based Q&A chatbot designed for querying research papers. It leverages LangChain, FAISS, and Groq's LLM (Gemma 2B) to extract relevant information from research documents efficiently. Users can input questions, and the chatbot retrieves the most relevant content using a Retrieval-Augmented Generation (RAG) approach.

## Features
- Load and Process PDFs: Reads multiple research papers from a directory.
-  Document Chunking: Splits PDFs into smaller text chunks for efficient retrieval.
-  FAISS-based Vector Store: Stores and retrieves document embeddings for similarity search.
-  LLM-powered Q&A: Uses Groq’s Gemma 2B to answer queries with context.
-  Fast Response Time: Measures query execution time for optimization.
-  Interactive UI: Built with Streamlit for an intuitive experience.

Installation & Setup
1. Install Dependencies
Ensure you have Python installed, then install the required packages:

```bash
pip install streamlit langchain langchain_community langchain_groq faiss-cpu python-dotenv
```
2. Set Up API Keys
Groq API Key is required. Store it in a .env file in the project directory:

- GROQ_API_KEY=your_api_key_here
Ensure the correct directory path is set for your PDFs.
3. Run the Application
Start the Streamlit app using:
```bash
streamlit run app.py
```
How to Use

- Upload PDFs: Place research papers in the specified directory.
- Embed Documents: Click "Document Embedding" to process and store document vectors.
- Ask Questions: Enter a query related to the research papers.
- View Results: The chatbot retrieves the most relevant response with document references.

How It Works

- Load Research Papers 📄 → Extracts text from PDFs.
- Text Splitting ✂ → Breaks documents into chunks for better retrieval.
- Generate Embeddings 🧠 → Uses Ollama Embeddings (Gemma 2B model).
- Store in FAISS 📊 → Creates a vector database for similarity search.
- Query Processing ❓ → Uses retrieval and LLM inference to provide accurate answers.
