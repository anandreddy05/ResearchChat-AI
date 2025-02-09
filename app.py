import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set the GROQ API key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")

# Initialize the language model
llm = ChatGroq(api_key=groq_api_key, model_name="gemma:2b")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        try:
            # Initialize embeddings and document loader with a specified model
            st.session_state.embeddings = OllamaEmbeddings(model="gemma:2b")  # Use the gemma model

            # Update the path to your directory containing PDFs
            directory_path = r"E:\Langchain Projects\1.Q&A Chatbot\RAG Q&A\research_papers"
            st.session_state.loader = PyPDFDirectoryLoader(directory_path)

            # Check if the directory exists and list the files
            if not os.path.exists(directory_path):
                st.write(f"Directory {directory_path} does not exist.")
                return
            
            files = os.listdir(directory_path)
            st.write(f"Files in directory: {files}")

            st.session_state.docs = st.session_state.loader.load()
            st.write(f"Loaded {len(st.session_state.docs)} documents.")
            
            # Split documents into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
            st.write(f"Split into {len(st.session_state.final_documents)} chunks.")

            # Check if documents contain text
            if not st.session_state.final_documents:
                st.write("No documents were found after splitting.")
                return

            # Debugging: Show an example document
            st.write("Example document content:", st.session_state.final_documents[0].page_content[:200])  # Print first 200 characters
            
            # Generate embeddings for the documents
            if hasattr(st.session_state.embeddings, "embed_documents"):
                embeddings = st.session_state.embeddings.embed_documents([doc.page_content for doc in st.session_state.final_documents])
            else:
                st.write("OllamaEmbeddings does not have the method 'embed_documents'. Please check the implementation.")
                return
            
            # Check the embeddings
            if not embeddings or any(len(e) == 0 for e in embeddings):
                st.write("Embeddings were not generated properly.")
                return
            
            # Create the FAISS vector store
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.write("Vector database initialized successfully.")
        except Exception as e:
            st.write(f"Error during vector initialization: {e}")  # Show error message


# Text input for user query
user_prompt = st.text_input("Enter your query from the research paper")

# Button to create the vector embeddings
if st.button("Document Embedding"):
    create_vector_embedding()

# Handling the user prompt
if user_prompt:
    if "vectors" in st.session_state:  # Check if vectors are initialized
        document_chain = create_stuff_documents_chain(llm,prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.process_time()  # Measure start time
        response = retrieval_chain.invoke({'input': user_prompt})
        end_time = time.process_time()  # Measure end time
        st.write(f"Response time: {end_time - start_time:.2f} seconds")  # Display elapsed time

        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('-------------------')
    else:
        st.write("Vector database is not initialized. Please click 'Document Embedding' first.")
