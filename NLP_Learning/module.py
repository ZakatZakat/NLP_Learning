from transformers import BertTokenizerFast
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
    
class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_lengt, qkv_bias=False, dropout=0):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_keys = nn.Linear(d_in, d_out)
        self.W_query = nn.Linear(d_in, d_out)
        self.W_values = nn.Linear(d_in, d_out)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)
        self.context_lengt = context_lengt
        self.register_buffer('mask', torch.triu(torch.ones(self.context_lengt, self.context_lengt), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_keys(x)
        querys = self.W_query(x)
        values = self.W_values(x)
        
        attention_scores = querys @ keys.transpose(-2, -1)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores = attention_scores.masked_fill(mask_bool, -float('inf'))
        
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attention_weights @ values
        


        context_vec = self.dropout(context_vec)
        context_vec = self.out_proj(context_vec)
        return context_vec
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
            ))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config['emb_dim'], 4 * config['emb_dim'])
        self.l2 = nn.Linear(4 * config['emb_dim'], config['emb_dim'])
        self.gelu = GELU()
    
    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], 
                                    cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'],
                                    cfg['emb_dim'])
        
        self.dropout = nn.Dropout(cfg['dropout'])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg)
                                          for _ in range(cfg['n_layers'])])
        
        self.final_norm = LayerNorm(cfg['emb_dim'])

        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds

        x = self.dropout(x)

        x = self.trf_blocks(x)
        #assert False, x

        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
                                    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiheadAttention(d_in = cfg['emb_dim'],
                                      d_out = cfg['emb_dim'],
                                      num_heads = cfg['num_head'],
                                      context_lengt = cfg['context_length'],
                                      dropout=cfg['dropout'], 
                                      qkv_bias=cfg['qkv_bias'])

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])

        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, x): 
        
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut

        return x
    
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.shift = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        normalized = (x - mean) / torch.sqrt(var + self.eps)

        return normalized * self.shift + self.bias


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


