import gradio as gr
import os
from rag import generate_answer  # Import functions from your rag.py

def clear_input():
    return "", ""

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
        
        query = gr.Textbox(
            label="Ask me anything about health and nutrition",
            placeholder="Example: What are the best sources of plant-based protein?",
            lines=3,
            elem_id="query-input"
        )
        
        answer = gr.Textbox(
            label="Expert Answer",
            placeholder="Your answer will appear here...",
            lines=8,
            elem_id="answer-output"
        )
        
        with gr.Row(elem_id="buttons-section"):
            submit_btn = gr.Button("Answer", elem_id="submit-btn")
            clear_btn = gr.Button("Clear", elem_id="clear-btn")
        
        # Connect the buttons to your RAG functions
        def generate_answer_with_error_handling(query):
            try:
                return generate_answer(query)
            except Exception as e:
                return f"An error occurred: {str(e)}"

        # Update the button click
        submit_btn.click(fn=generate_answer_with_error_handling, inputs=query, outputs=answer)
        clear_btn.click(fn=lambda: ("", ""), outputs=[query, answer])

# Launch the interface
if __name__ == "__main__":
    interface.launch(share=True)