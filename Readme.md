# RAG Chatbot using Flask, Weaviate, and LangChain

## Overview
This is a **Retrieval-Augmented Generation (RAG) Chatbot** built using Flask, Weaviate, and LangChain. The chatbot allows users to upload documents, extract embeddings, store them in a vector database, and retrieve information using an LLM-based chat system. The chatbot uses **Together API** for language model inference and **Weaviate** as the vector database.

## Features
- Upload and process PDF documents for knowledge extraction.
- Store document chunks in **Weaviate vector database**.
- Use **LangChain** to retrieve context-based answers.
- Support for **Together AI** LLM models.
- Maintain conversation history in a **MySQL** database.
- Flask-based web interface for chat interactions.

## Technologies Used
- **Backend**: Flask
- **Database**: MySQL
- **Vector Store**: Weaviate
- **LLM Integration**: LangChain + Together API
- **Embeddings**: HuggingFace Sentence-Transformers
- **PDF Processing**: PyPDFLoader
- **Frontend**: HTML (via Flask templates)

---

## Installation & Setup

### Prerequisites
- Python **3.8+**
- MySQL database installed and configured
- Weaviate cloud instance
- Together API key

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Environment Variables
Set up environment variables in a `.env` file or export them:
```sh
export DB_HOST='your_db_host'
export DB_USER='your_db_user'
export DB_PASSWORD='your_db_password'
export DB_NAME='rag_chatbot_db'
```

### Database Setup
Run the following SQL script to create the necessary tables:
```sql
CREATE DATABASE rag_chatbot_db;
USE rag_chatbot_db;

CREATE TABLE uploaded_documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    weaviate_url TEXT,
    chunk_size INT,
    chunk_overlap INT,
    model_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE conversation_threads (
    id INT AUTO_INCREMENT PRIMARY KEY,
    document_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES uploaded_documents(id)
);

CREATE TABLE messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    thread_id INT,
    role ENUM('user', 'assistant') NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (thread_id) REFERENCES conversation_threads(id)
);
```

### Running the Flask App
```sh
python app.py
```
The app will run on [http://localhost:5000/](http://localhost:5000/).

---

## API Endpoints

### `/config` (GET/POST)
Handles API key validation, file upload, and system configuration.
- **GET**: Renders the configuration page.
- **POST**: Processes API keys, saves uploaded files, initializes Weaviate, and redirects to the chat page.

### `/chat` (GET/POST)
Handles user interactions and chat history retrieval.
- **GET**: Displays previous chat messages.
- **POST**: Processes user input, retrieves context, generates responses, and saves messages.

---

## How It Works
1. User uploads a document on the `/config` page.
2. Text chunks are extracted and embedded using HuggingFace models.
3. Chunks are stored in **Weaviate**.
4. User asks a question on the `/chat` page.
5. Context is retrieved from **Weaviate**.
6. A response is generated using **Together AI LLM**.
7. Chat history is saved in **MySQL**.

---

## Error Handling & Logging
- Errors are logged using Python’s `logging` module.
- Connection failures, invalid API keys, and missing files are handled with appropriate error messages.
- Debugging logs can be enabled in the `initialize_rag_system` function.

---

## Future Enhancements
- Support for multiple document formats (**TXT, DOCX**, etc.).
- UI enhancements using **Bootstrap** or **React**.
- Streaming LLM responses for better user experience.
- User authentication and access control.

---

## Author
Developed by **Shiva**.

