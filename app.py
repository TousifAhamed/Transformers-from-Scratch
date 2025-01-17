import gradio as gr
from model_utils import load_model, generate_text

# Load the model globally
model_path = "path/to/your/downloaded/best_model.pt"  # Update this path
model = load_model(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def predict(prompt, max_tokens, temperature, top_k):
    """Wrapper function for Gradio interface"""
    generated_text = generate_text(
        model=model,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k
    )
    return generated_text

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Enter your prompt", lines=3),
        gr.Slider(minimum=1, maximum=200, value=50, step=1, label="Max Tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Top-k"),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=5),
    title="GPT Text Generation",
    description="Enter a prompt and the model will generate text continuation.",
    examples=[
        ["The quick brown fox", 50, 0.8, 40],
        ["Once upon a time", 100, 0.9, 50],
        ["In the distant future", 75, 0.7, 30],
    ],
)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)  # set share=False if you don't want to share publicly 