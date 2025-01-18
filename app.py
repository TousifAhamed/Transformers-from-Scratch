import gradio as gr
import torch
import os
from model_utils import load_model, generate_text

# Initialize model
try:
    model_path = "best_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    model = load_model(model_path)
    if model is None:
        raise ValueError("Model failed to load")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict(prompt, max_tokens, temperature, top_k):
    """Wrapper function for Gradio interface"""
    if not prompt:
        return "Error: Please enter a prompt"
    if model is None:
        return "Error: Model failed to load. Please check the logs."
    
    try:
        generated_text = generate_text(
            model=model,
            prompt=prompt.strip(),
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_k=int(top_k)
        )
        return generated_text
    except Exception as e:
        return f"Error during generation: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Enter your prompt", lines=3, placeholder="Type your text here..."),
        gr.Slider(minimum=1, maximum=200, value=50, step=1, label="Max Tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Top-k"),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=5),
    title="GPT Text Generation",
    description="Enter a text prompt and the model will generate a continuation.",
    examples=[
        ["The quick brown fox", 50, 0.8, 40],
        ["Once upon a time", 100, 0.9, 50],
        ["In the distant future", 75, 0.7, 30],
    ],
)

if __name__ == "__main__":
    demo.launch(share=False)
else:
    app = demo.launch(share=False)