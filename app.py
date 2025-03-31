# Flask-related imports
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename

# Standard library imports
import os
import logging
import atexit
from pathlib import Path

# Database-related imports
import pymysql
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Weaviate and LangChain integrations
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# LangChain LLMs and utilities
from langchain.prompts import ChatPromptTemplate
from langchain_together import ChatTogether
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from together import Together  # Direct import for Together AI


# -----------------------------
# Database Configuration
# -----------------------------

import os
import logging
from pathlib import Path
from flask import Flask

# Use environment variables for security instead of hardcoded credentials
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),  # Ensure this is securely stored
    'password': os.getenv('DB_PASSWORD', 'your_secure_password'),  # Replace with a secure method
    'database': os.getenv('DB_NAME', 'rag_chatbot_db'),
}

# -----------------------------
# Flask App Configuration
# -----------------------------

app = Flask(__name__)

# Generate a secure secret key for session management
app.secret_key = os.urandom(24)

# Configure upload settings
app.config['UPLOAD_FOLDER'] = 'temp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

# Ensure the upload folder exists
upload_path = Path(app.config['UPLOAD_FOLDER'])
upload_path.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Logging Setup
# -----------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Database Utility Functions
# -----------------------------

def get_db_connection():
    """Establish a connection to the MySQL database."""
    try:
        connection = pymysql.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database'],
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

def save_uploaded_document(filename, original_filename, file_path, config):
    """
    Save details of an uploaded document into the database.
    
    Args:
        filename (str): System-generated filename.
        original_filename (str): User-uploaded filename.
        file_path (str): Path where the file is stored.
        config (dict): Weaviate and chunking configuration.

    Returns:
        int: Document ID of the inserted record.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO uploaded_documents 
            (filename, original_filename, file_path, weaviate_url, chunk_size, chunk_overlap, model_name) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                filename, 
                original_filename, 
                str(file_path), 
                config.get('weaviate_url', ''), 
                config.get('chunk_size', 1000), 
                config.get('chunk_overlap', 200), 
                config.get('model_name', '')
            ))
        conn.commit()
        return cursor.lastrowid
    except Exception as e:
        logger.error(f"Error saving document: {e}")
        raise
    finally:
        conn.close()

def create_conversation_thread(document_id):
    """
    Create a new conversation thread linked to a document.
    
    Args:
        document_id (int): ID of the uploaded document.

    Returns:
        int: Thread ID of the inserted conversation.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO conversation_threads (document_id) VALUES (%s)"
            cursor.execute(sql, (document_id,))
        conn.commit()
        return cursor.lastrowid
    except Exception as e:
        logger.error(f"Error creating conversation thread: {e}")
        raise
    finally:
        conn.close()

def save_message(thread_id, role, content):
    """
    Save a message into a conversation thread.

    Args:
        thread_id (int): ID of the conversation thread.
        role (str): Sender role (e.g., 'user' or 'assistant').
        content (str): Message text.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO messages (thread_id, role, content) VALUES (%s, %s, %s)"
            cursor.execute(sql, (thread_id, role, content))
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving message: {e}")
        raise
    finally:
        conn.close()

def get_conversation_history(thread_id):
    """
    Retrieve all messages from a given conversation thread.

    Args:
        thread_id (int): ID of the conversation thread.

    Returns:
        list: List of messages with role, content, and timestamp.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
            SELECT role, content, timestamp 
            FROM messages 
            WHERE thread_id = %s 
            ORDER BY timestamp
            """
            cursor.execute(sql, (thread_id,))
            return cursor.fetchall()
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        raise
    finally:
        conn.close()

# -----------------------------
# Cleanup Function
# -----------------------------

def cleanup():
    """Delete temporary files when the application exits."""
    if not app.config.get('TESTING', False):  # Prevent cleanup during testing
        for file in upload_path.glob('*'):
            try:
                file.unlink()
                logger.info(f"Deleted temporary file: {file}")
            except Exception as e:
                logger.error(f"Error deleting file {file}: {e}")

atexit.register(cleanup)


import traceback
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_together import ChatTogether
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def initialize_rag_system(config):
    """
    Initialize the RAG system components with enhanced error handling.

    Args:
        config (dict): Configuration containing Weaviate, Together API keys, file paths, and model settings.

    Returns:
        tuple: (vector_db, rag_chain) - The initialized vector database and retrieval-augmented generation chain.
    
    Raises:
        RuntimeError: If initialization fails due to missing configurations or API failures.
    """
    try:
        logger.info("=" * 50)
        logger.info("Starting RAG System Initialization")
        logger.info("=" * 50)

        # -----------------------------
        # Configuration Validation
        # -----------------------------

        required_keys = ['weaviate_url', 'weaviate_api_key', 'together_api_key', 'uploaded_file']
        missing_keys = [key for key in required_keys if not config.get(key)]

        if missing_keys:
            logger.error(f"Missing configuration keys: {missing_keys}")
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

        # Basic validation for API keys and URLs
        if not config['weaviate_url'].startswith("http"):
            raise ValueError("Invalid Weaviate URL. Ensure it starts with 'http' or 'https'.")
        if len(config['weaviate_api_key']) < 10:
            raise ValueError("Invalid Weaviate API Key.")
        if len(config['together_api_key']) < 10:
            raise ValueError("Invalid Together API Key.")

        # -----------------------------
        # Connect to Weaviate
        # -----------------------------

        logger.info("Connecting to Weaviate...")
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=config['weaviate_url'],
            auth_credentials=Auth.api_key(config['weaviate_api_key'])
        )
        logger.info("Weaviate connection successful.")

        # -----------------------------
        # Load and Process Documents
        # -----------------------------

        logger.info("Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        logger.info("Processing PDF document...")
        loader = PyPDFLoader(str(config['uploaded_file']))
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get('chunk_size', 1000),
            chunk_overlap=config.get('chunk_overlap', 200)
        )
        docs = text_splitter.split_documents(pages)

        # -----------------------------
        # Create Vector Store
        # -----------------------------

        logger.info("Creating vector store in Weaviate...")
        vector_db = WeaviateVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            client=client,
            index_name="DocumentChunks"
        )

        # -----------------------------
        # Initialize Language Model
        # -----------------------------

        logger.info("Initializing LLM with Together API...")
        llm = ChatTogether(
            together_api_key=config['together_api_key'],
            model=config.get('model_name', 'mistral-7b'),
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 512)
        )

        # -----------------------------
        # Setup RAG Chain
        # -----------------------------

        template = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use ten sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": vector_db.as_retriever(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("RAG system initialized successfully.")
        return vector_db, rag_chain

    except Exception as e:
        logger.error(f"RAG Initialization Failed: {str(e)}")
        logger.debug(traceback.format_exc())  # Detailed stack trace for debugging
        raise RuntimeError(f"Comprehensive Initialization Failure: {str(e)}")



# ==========================================================================
# Configuration Route (/config) - Handles API Key Validation & File Uploads
# ==========================================================================

@app.route('/config', methods=['GET', 'POST'])
def config():
    """
    Handles the configuration of the RAG system, including API key validation, 
    file upload handling, and database initialization.
    
    Returns:
        - On GET: Renders the configuration page.
        - On POST: Processes the configuration and redirects to the chat page.
    """
    if request.method == 'POST':
        try:
            # Extract and sanitize form inputs
            weaviate_url = request.form['weaviate_url'].strip()
            weaviate_api_key = request.form['weaviate_api_key'].strip()
            together_api_key = request.form['together_api_key'].strip()
            chunk_size = max(100, min(int(request.form['chunk_size']), 2000))
            chunk_overlap = max(0, min(int(request.form['chunk_overlap']), 500))
            model_name = request.form['model_name'].strip()
            temperature = max(0.1, min(float(request.form['temperature']), 1.0))
            max_tokens = max(100, min(int(request.form['max_tokens']), 2000))

            logger.info("Received configuration data from user.")

            # Validate required API keys
            if not all([weaviate_url, weaviate_api_key, together_api_key]):
                raise ValueError("Missing required API keys.")

            # Handle file upload
            uploaded_file = request.files.get('document')
            if not uploaded_file or uploaded_file.filename == '':
                raise ValueError("No document uploaded.")

            filename = secure_filename(uploaded_file.filename)
            file_path = upload_path / filename
            uploaded_file.save(file_path)

            logger.info(f"Uploaded document saved at: {file_path}")

            # Save document to the database and get document ID
            document_id = save_uploaded_document(
                filename=filename,
                original_filename=uploaded_file.filename,
                file_path=file_path,
                config={
                    'weaviate_url': weaviate_url,
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'model_name': model_name
                }
            )

            # Create a conversation thread
            thread_id = create_conversation_thread(document_id)

            # Store configuration in session
            session.update({
                'document_id': document_id,
                'thread_id': thread_id,
                'weaviate_url': weaviate_url,
                'weaviate_api_key': weaviate_api_key,
                'together_api_key': together_api_key,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'model_name': model_name,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'uploaded_filename': filename
            })

            logger.info(f"Session updated with document ID: {document_id}, thread ID: {thread_id}")

            # Initialize RAG system
            vector_db, rag_chain = initialize_rag_system({
                **session,
                'uploaded_file': file_path
            })

            # Save initial assistant message
            initial_message = "Hello! I'm your RAG assistant. How can I help you today?"
            save_message(thread_id, 'assistant', initial_message)

            logger.info("RAG system initialized successfully. Redirecting to chat.")

            return redirect(url_for('chat'))

        except Exception as e:
            logger.error(f"Configuration error: {e}")
            return render_template('config.html', error=str(e))

    return render_template('config.html')


# ======================================================================
# Chat Route (`/chat`) - Handles user interactions with the RAG system.
# ====================================================================

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """Handles user queries and responses in the chat interface."""
    
    # Redirect user to config page if thread_id is not in session
    if 'thread_id' not in session:
        return redirect(url_for('config'))
    
    thread_id = session['thread_id']  # Retrieve thread ID from session

    if request.method == 'POST':
        try:
            # Get user prompt from form
            prompt = request.form.get('prompt', '').strip()
            if not prompt:
                raise ValueError("Please enter a question")
            
            # Save user message to database
            save_message(thread_id, 'user', prompt)

            # Reinitialize RAG system for generating responses
            _, rag_chain = initialize_rag_system({
                'uploaded_file': upload_path / session['uploaded_filename'],
                'document_id': session["document_id"],
                'thread_id': thread_id,
                'weaviate_url': session["weaviate_url"],
                'weaviate_api_key': session["weaviate_api_key"],
                'together_api_key': session["together_api_key"],
                'chunk_size': session["chunk_size"],
                'chunk_overlap': session["chunk_overlap"],
                'model_name': session['model_name'],
                'temperature': session['temperature'],
                'max_tokens': session['max_tokens']                
            })
            
            # Generate response using RAG system
            response = rag_chain.invoke(prompt)

            # Save assistant response to database
            save_message(thread_id, 'assistant', response)

        except Exception as e:
            logger.error(f"Chat error: {e}")
            response = f"Error: {e}"
            save_message(thread_id, 'assistant', response)
    
    # Retrieve complete conversation history for display
    messages = get_conversation_history(thread_id)
    
    # Render chat UI with messages
    return render_template('chat.html', messages=messages)

# ==========================================
# Flask App Runner
# ==========================================

if __name__ == '__main__':
    app.run(port=5005)
