import torch
import torch.nn.functional as F
from dataclasses import dataclass
import tiktoken

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.05

class GPT:
    # ... [Your original GPT class implementation] ...
    pass

def load_model(model_path):
    """Load the trained model"""
    checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def generate_text(model, prompt, max_new_tokens=50, temperature=0.8, top_k=40):
    """Generate text based on a prompt"""
    enc = tiktoken.get_encoding("gpt2")
    input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)
    
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.size(1) > model.config.block_size:
                input_ids = input_ids[:, -model.config.block_size:]
            
            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == enc.encode_single_token("<|endoftext|>"):
                break
    
    generated_text = enc.decode(input_ids[0].tolist())
    return generated_text 