from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import logging
from logging.handlers import RotatingFileHandler
import time

# Initialize Flask app
app = Flask(__name__)

# Configure logging for production
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/medicalbot.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('MedicalBot startup')

# Load environment variables
load_dotenv()

# Get API keys from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate required environment variables
if not PINECONE_API_KEY:
    app.logger.error("PINECONE_API_KEY not found in environment variables")
if not GROQ_API_KEY:
    app.logger.warning("GROQ_API_KEY not found in environment variables")

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY or ""

# Initialize embeddings and vector store with error handling
try:
    embeddings = download_hugging_face_embeddings()
    index_name = "medicalbot"
    
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    app.logger.info("Pinecone vector store initialized successfully")
except Exception as e:
    app.logger.error(f"Failed to initialize Pinecone vector store: {e}")
    # Set to None to handle gracefully in the chat function
    retriever = None

# Advanced prompt: add instructions for empathy and clarity
advanced_system_prompt = (
    "You are HealthMate, an advanced AI medical assistant. "
    "Provide clear, empathetic, and concise answers based on the retrieved context. "
    "If unsure, say you don't know and recommend consulting a healthcare professional. "
    "Limit your answer to three sentences. Use layman's terms when possible.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", advanced_system_prompt),
        ("human", "{input}"),
    ]
)

# List of available Groq models to try in order
GROQ_MODELS = [
    "llama-3.1-8b-instant",  # Newer, faster model
    "llama-3.1-70b-versatile",  # More capable model
    "llama3-70b-8192",  # Alternative model
    "llama3-8b-8192",   # Another alternative
]

# Clean Groq chain creation with robust error handling
def get_groq_chain(model_name="llama-3.1-8b-instant"):
    try:
        from langchain_groq import ChatGroq
        
        if not GROQ_API_KEY:
            app.logger.warning("GROQ_API_KEY not found in environment")
            return None
            
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=0.4,
            max_tokens=500,
            timeout=30  # Add timeout for production
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)
    except ImportError:
        app.logger.error("langchain-groq package not installed. Run: pip install langchain-groq")
        return None
    except Exception as e:
        app.logger.error(f"Groq chain creation failed for model {model_name}: {e}")
        return None

# Simple fallback responses for common medical questions
def get_fallback_response(question):
    question_lower = question.lower()
    
    # Simple pattern matching for common medical questions
    if any(word in question_lower for word in ['fever', 'temperature']):
        return "For fever, rest and hydration are important. If temperature is above 103°F (39.4°C) or persists for more than 3 days, consult a doctor."
    
    elif any(word in question_lower for word in ['headache', 'migraine']):
        return "For headaches, try resting in a quiet, dark room. Stay hydrated and consider over-the-counter pain relief if appropriate. If headaches are severe or frequent, see a doctor."
    
    elif any(word in question_lower for word in ['cold', 'flu', 'cough']):
        return "For cold or flu symptoms, rest, hydration, and over-the-counter remedies can help. If symptoms persist beyond 10 days or include high fever, seek medical advice."
    
    elif any(word in question_lower for word in ['pain', 'hurt']):
        return "For pain management, rest the affected area and consider appropriate pain relief. If pain is severe, persistent, or accompanied by other symptoms, please consult a healthcare professional."
    
    else:
        return "I'm currently experiencing technical difficulties. For medical advice, please consult a healthcare professional directly."

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        app.logger.info(f"User input: {msg}")

        # Check if vector store is initialized
        if retriever is None:
            app.logger.error("Vector store not initialized")
            return get_fallback_response(msg)

        # Try Groq with multiple models
        app.logger.info("Attempting Groq API...")
        for model in GROQ_MODELS:
            groq_chain = get_groq_chain(model)
            if groq_chain:
                try:
                    # Add timeout to prevent hanging requests
                    response = groq_chain.invoke({"input": msg})
                    if response and "answer" in response:
                        app.logger.info(f"Groq response successful with model {model}")
                        return str(response["answer"])
                except Exception as groq_e:
                    app.logger.warning(f"Groq invocation failed with model {model}: {groq_e}")
                    continue

        # Final fallback: simple pattern matching
        app.logger.info("Using fallback response system")
        fallback_response = get_fallback_response(msg)
        return fallback_response
        
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {e}")
        return "I'm experiencing technical difficulties. Please try again later or consult a healthcare professional directly."

# Health check endpoint for deployment platforms
@app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    # Never run in debug mode in production
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
