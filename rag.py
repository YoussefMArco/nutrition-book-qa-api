import os
import signal
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr

# Load API key from environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
    context = ""
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context

def generate_answer(query):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name='gemini-pro')
    context = get_relevant_context_from_db(query)
    prompt = generate_rag_prompt(query=query, context=context)
    answer = model.generate_content(prompt)
    return answer.text

def clear_input():
    return "", ""

# Custom CSS for a polished health-themed UI
custom_css = """
    .gradio-container {
        background: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)),
                    url('https://images.unsplash.com/photo-1490645935967-10de6ba17061?auto=format&fit=crop&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    #container {
        max-width: 850px;
        margin: 2rem auto;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    #header {
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        padding: 1.5rem;
        background: linear-gradient(135deg, #43a047, #1b5e20);
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    #header h1 {
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 1px 2px 3px rgba(0, 0, 0, 0.2);
    }
    #header h3 {
        font-size: 1.2rem;
        font-weight: 400;
        opacity: 0.8;
    }
    #buttons-section {
        display: flex;
        justify-content: space-between;
        margin-top: 1rem;
    }
    #submit-btn, #clear-btn {
        width: 48%;
        height: 45px;
        border-radius: 25px;
        font-weight: 500;
        transition: transform 0.2s;
    }
    #submit-btn {
        background: linear-gradient(135deg, #43a047, #1b5e20);
        color: white;
        border: none;
    }
    #clear-btn {
        background: #f5f5f5;
        border: 2px solid #43a047;
        color: #333;
    }
    #submit-btn:hover, #clear-btn:hover {
        transform: translateY(-2px);
    }
    @media (max-width: 768px) {
        #container {
            padding: 1rem;
        }
        #header {
            padding: 1rem;
        }
        #buttons-section {
            flex-direction: column;
            gap: 1rem;
        }
        #submit-btn, #clear-btn {
            width: 100%;
        }
    }
"""

# Gradio interface
with gr.Blocks(css=custom_css) as interface:
    with gr.Column(elem_id="container"):
        with gr.Column(elem_id="header"):
            gr.Markdown("""# 🌿 Health & Nutrition Assistant
            ### Your AI-powered healthcare knowledge companion""")
        query = gr.Textbox(label="Ask me anything about health and nutrition",
                           placeholder="Example: What are the best sources of plant-based protein?",
                           lines=3, elem_id="query-input")
        answer = gr.Textbox(label="Expert Answer",
                            placeholder="Your answer will appear here...",
                            lines=8, elem_id="answer-output")
        with gr.Row(elem_id="buttons-section"):
            submit_btn = gr.Button("Ask Question", elem_id="submit-btn")
            clear_btn = gr.Button("Clear", elem_id="clear-btn")

    submit_btn.click(fn=generate_answer, inputs=query, outputs=answer)
    clear_btn.click(fn=clear_input, inputs=None, outputs=[query, answer])

# Run the application
if __name__ == "__main__":
    print("Choose a mode:")
    print("1: Gradio Web Interface")
    choice = input("Enter your choice (1): ").strip()
    if choice == "1":
        interface.queue()
        interface.launch(share=True)
    else:
        print("Invalid choice. Exiting.")
