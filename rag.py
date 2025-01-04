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

# Functions for RAG prompt generation and database query
def generate_rag_prompt(query, context):
    escaped = context.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""
You are a helpful and informative bot that answers questions using text from the reference context included below. 
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
strike a friendly and conversational tone. 
If the context is irrelevant to the answer, you may ignore it.
QUESTION: '{query}'
CONTEXT: '{context}'
ANSWER:
"""
    return prompt

def get_relevant_context_from_db(query):
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)
        search_results = vector_db.similarity_search(query, k=6)
        context = "\n".join([result.page_content for result in search_results])
        sources = list(set([result.metadata.get("source", "Unknown") for result in search_results]))  # Remove duplicates
        logger.info(f"Retrieved context: {context}")
        logger.info(f"Sources: {sources}")
        return context, sources
    except Exception as e:
        logger.error(f"Error retrieving context from database: {e}", exc_info=True)
        return "", []

def generate_answer(query):
    try:
        logger.info(f"Generating answer for query: {query}")
        
        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(model_name='gemini-pro')
        
        # Retrieve context and sources from Chroma DB
        context, sources = get_relevant_context_from_db(query)
        logger.info(f"Retrieved context: {context}")
        logger.info(f"Sources: {sources}")
        
        # Generate prompt
        prompt = generate_rag_prompt(query=query, context=context)
        logger.info(f"Generated prompt: {prompt}")
        
        # Generate answer using Gemini
        answer = model.generate_content(prompt)
        logger.info(f"Generated answer: {answer.text}")
        
        sources.sort()
        
        # Format the answer with sources
        formatted_answer = f"{answer.text}\n\nSources:\n" + "\n".join([f'- {source.replace(".pdf", "")}' for source in sources])
        return formatted_answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        return f"An error occurred while generating the answer. Please try again. Error: {str(e)}"