Medical Chatbot with Groq API

A medical question-answering chatbot that uses Groq's language models for response generation and Pinecone for vector storage and retrieval. This application processes medical PDF documents to create a knowledge base and provides conversational interface for medical queries.

Features

Document Processing: Load and chunk PDF documents for efficient retrieval

Semantic Search: Uses HuggingFace embeddings and Pinecone vector storage

Groq Integration: Leverages multiple Groq language models for response generation

Fallback System: Intelligent fallback responses for common medical questions

Web Interface: Clean Flask-based chat interface

Prerequisites
Before you begin, ensure you have the following:

Python 3.10 or higher

Conda package manager

Pinecone API account (sign up here)

Groq API account (sign up here)

Installation
1. Clone the Repository
bash
git clone <your-repository-url>
cd End-to-end-Medical-Chatbot-Generative-AI
2. Create Conda Environment
bash
conda create -n medibot python=3.10 -y
conda activate medibot
3. Install Dependencies
bash
pip install -r requirements.txt
4. Environment Configuration
Create a .env file in the root directory with your API credentials:

ini
PINECONE_API_KEY="your_pinecone_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"
Setting Up Pinecone
Create a Pinecone account at pinecone.io

Get your API key from the Pinecone console

Create an index named "medicalbot" with:

Dimension: 384

Metric: cosine

Cloud: AWS

Region: us-east-1

Preparing Your Data
Create a Data/ directory in your project root

Place your medical PDF files in the Data/ directory

The application will process these files to create the knowledge base

Running the Application
1. Process Documents and Create Embeddings
bash
python store_index.py
This command will:

Load PDF files from the Data/ directory

Split them into text chunks

Generate embeddings using HuggingFace's all-MiniLM-L6-v2 model

Store the embeddings in your Pinecone index

2. Start the Chatbot Application
bash
python app.py
3. Access the Application
Open your web browser and navigate to:

text
http://localhost:8080
Usage
Type your medical question in the chat interface

The system will:

Search for relevant context in your PDF documents

Generate a response using Groq's language models

Provide a concise, medically-informed answer

If technical issues occur, the system will provide appropriate fallback responses

The application tries multiple Groq models in sequence for optimal performance:

llama-3.1-8b-instant (primary)

llama-3.1-70b-versatile (fallback)

llama3-70b-8192 (secondary fallback)

llama3-8b-8192 (final fallback)
# DCOchatbot
