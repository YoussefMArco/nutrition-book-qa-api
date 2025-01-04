import gradio as gr
import os
from rag import generate_chat_answer, conversation  # Import updated functions
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_chat():
    conversation.clear_history()
    return "", []

# Define custom CSS
custom_css = """
.gradio-container {
    background: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)),
                url('file:///D:/Huawei NTI Ai Tr aining/Projecttt/Background.png');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

/* Add rounded corners and shadow to the main container */
#container {
    background: white;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 20px;
    margin: 20px;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
}

/* Style the header */
#header {
    background: #4CAF50;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
    color: white;
}

/* Style the buttons */
#buttons-section {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-top: 20px;
}

#submit-btn {
    background: #4CAF50 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 10px 20px !important;
    min-width: 120px;
}

#clear-btn {
    background: white !important;
    color: #4CAF50 !important;
    border: 1px solid #4CAF50 !important;
    border-radius: 5px !important;
    padding: 10px 20px !important;
    min-width: 120px;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css) as interface:
    with gr.Column(elem_id="container"):
        with gr.Column(elem_id="header"):
            gr.Markdown("""
            # 🌿 Health & Nutrition Assistant 
            ### Your AI-powered healthcare knowledge companion
            """)
        
        chatbot = gr.Chatbot(label="Chat History", elem_id="chatbot", type="messages")
        query = gr.Textbox(
            label="Ask me anything about health and nutrition",
            placeholder="Example: What are the best sources of plant-based protein?",
            lines=3,
            elem_id="query-input"
        )
        
        with gr.Row(elem_id="buttons-section"):
            submit_btn = gr.Button("Send", elem_id="submit-btn")
            clear_btn = gr.Button("Clear Chat", elem_id="clear-btn")
        
        # Connect the buttons to your RAG functions
        def respond(query, chat_history):
            if not query.strip():
                return "Please enter a valid question.", chat_history
            
            answer = generate_chat_answer(query)
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer})
            logger.info(f"Updated Chat History:\n{chat_history}")  # Log the updated chat history
            return "", chat_history

        # Update the button click
        submit_btn.click(fn=respond, inputs=[query, chatbot], outputs=[query, chatbot])
        clear_btn.click(fn=clear_chat, outputs=[query, chatbot])

# Launch the interface
if __name__ == "__main__":
    interface.launch(share=True)