import torch
import torch.nn as nn
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

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        assert self.n_embd % self.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1))))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.c_fc(x))
        x = self.dropout(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}."

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), label_smoothing=0.05)
        return logits, loss

def load_model(model_path):
    """Load the trained model"""
    try:
        torch.serialization.add_safe_globals({'GPTConfig': GPTConfig})
        checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        config = GPTConfig(**checkpoint['config'])
        model = GPT(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def generate_text(model, prompt, max_new_tokens=50, temperature=0.8, top_k=40):
    """Generate text based on a prompt
    Args:
        model: The GPT model
        prompt (str): Input text to continue from
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Higher values produce more diverse text (default: 0.8)
        top_k (int): Number of highest probability tokens to consider (default: 40)
    Returns:
        str: Generated text including the original prompt
    """
    try:
        # Initialize tokenizer and encode prompt
        enc = tiktoken.get_encoding("gpt2")
        input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0)
        
        # Move to same device as model
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Generate tokens
        with torch.no_grad():
            generated_tokens = []
            for _ in range(max_new_tokens):
                # Truncate if sequence length exceeds block size
                if input_ids.size(1) > model.config.block_size:
                    input_ids = input_ids[:, -model.config.block_size:]
                
                # Get predictions from model
                logits, _ = model(input_ids)
                logits = logits[:, -1, :]  # Get last token's logits
                
                # Apply temperature scaling
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append the token and continue generating
                generated_tokens.append(next_token.item())
                input_ids = torch.cat((input_ids, next_token), dim=1)

            # Decode the generated tokens
            output_text = prompt + enc.decode(generated_tokens)
            return output_text

    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        return prompt

