import os
from dotenv import load_dotenv
import signal
import sys
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key from environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

# Graceful shutdown on SIGINT
def signal_handler(sig, frame):
    print('\nThanks for using Gemini. :)')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Conversation class to manage chat history
class Conversation:
    def __init__(self):
        self.history = []

    def add_user_message(self, message):
        self.history.append({"role": "user", "content": message})

    def add_bot_message(self, message):
        self.history.append({"role": "assistant", "content": message})

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []

# Initialize a global conversation object
conversation = Conversation()

# Function to refine the query using conversation history
def refine_query_with_history(query, history):
    """
    Refines the query by incorporating conversation history.
    """
    # Format the conversation history
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    
    # Create a prompt to refine the query
    refine_prompt = f"""
You are a helpful assistant that refines user queries based on conversation history. 
Your task is to make the query more specific and relevant by incorporating context from the conversation history.

CONVERSATION HISTORY:
{history_str}

USER QUERY: '{query}'

REFINED QUERY:
"""
    # Use Gemini to generate the refined query
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name='gemini-pro')
    response = model.generate_content(refine_prompt)
    refined_query = response.text.strip()
    
    logger.info(f"Original Query: {query}")
    logger.info(f"Refined Query: {refined_query}")
    
    return refined_query

# Functions for RAG prompt generation and database query
def generate_chat_prompt(query, context, history):
    # Format the conversation history
    if not history:
        history_str = "No prior conversation history."
    else:
        # Format the conversation history
        history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
        
    logger.info(f"Conversation History:\n{history_str}")  # Log the conversation history
    logger.info(f"Retrieved Context:\n{context}")  # Log the retrieved context
    
    prompt = f"""
You are a helpful and informative chatbot that answers questions using text from the reference context included below. 
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
strike a friendly and conversational tone. 
If the context is irrelevant to the answer, you may ignore it.

CONVERSATION HISTORY:
{history_str}

CONTEXT:
{context}

USER QUESTION: '{query}'
BOT ANSWER:
"""
    return prompt

def get_relevant_context_from_db(query, history):
    """
    Retrieves relevant context from ChromaDB using a refined query.
    """
    try:
        # Refine the query using conversation history
        refined_query = refine_query_with_history(query, history)
        
        # Retrieve context using the refined query
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)
        search_results = vector_db.similarity_search(refined_query, k=6)
        context = "\n".join([result.page_content for result in search_results])
        sources = list(set([result.metadata.get("source", "Unknown") for result in search_results]))  # Remove duplicates
        
        logger.info(f"Refined Query: {refined_query}")
        logger.info(f"Retrieved Context: {context}")
        logger.info(f"Sources: {sources}")
        
        return context, sources
    except Exception as e:
        logger.error(f"Error retrieving context from database: {e}", exc_info=True)
        return "", []

def generate_chat_answer(query):
    try:
        logger.info(f"Generating answer for query: {query}")
        
        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name='gemini-pro')
        
        # Retrieve context and sources from Chroma DB using refined query
        context, sources = get_relevant_context_from_db(query, conversation.get_history())
        logger.info(f"Retrieved context: {context}")
        logger.info(f"Sources: {sources}")
        
        # Generate prompt with conversation history
        prompt = generate_chat_prompt(query, context, conversation.get_history())
        logger.info(f"Generated prompt: {prompt}")
        
        # Generate answer using Gemini
        answer = model.generate_content(prompt)
        logger.info(f"Generated answer: {answer.text}")
        
        # Add user query and bot response to conversation history
        conversation.add_user_message(query)
        conversation.add_bot_message(answer.text)
        
        # Format the answer with cleaned and sorted sources
        formatted_answer = f"{answer.text}\n\nSources:\n" + "\n".join([f'- {source.replace(".pdf", "")}' for source in sorted(sources)])
        
        return formatted_answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        return f"An error occurred while generating the answer. Please try again. Error: {str(e)}"