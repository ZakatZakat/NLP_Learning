from transformers import BertTokenizerFast
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


def read_csv(data=None, Target=None):
        # Try with ISO-8859-1
    try:
        data = pd.read_csv('/home/oskar/NLP/datasets/Corona_NLP_train.csv', encoding='ISO-8859-1')
        print("File read successfully with ISO-8859-1 encoding.")
    except UnicodeDecodeError as e:
        print(f"Failed to read file with ISO-8859-1: {e}")

    # If the above fails, try with Windows-1252
    try:
        data = pd.read_csv('/home/oskar/NLP/datasets/Corona_NLP_train.csv', encoding='cp1252')
        print("File read successfully with Windows-1252 encoding.")
    except UnicodeDecodeError as e:
        print(f"Failed to read file with Windows-1252: {e}")

    data['Target'] = data['Sentiment'].replace({value:key for key, value in enumerate(data['Sentiment'].unique())})
    return data

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        values = self.texts[idx]
        label = self.labels[idx]

        embedded = self.tokenizer.encode_plus(
                     values,
                     add_special_tokens=True,
                     max_length=520,
                     padding='max_length',
                     truncation=True,
                     return_attention_mask=True,
                     return_tensors='pt'
                    )
        input_ids = embedded['input_ids'].flatten()
        masks = embedded['attention_mask'].flatten()

        return {'input_ids': input_ids, 'masks': masks, 'labels': torch.tensor(label, dtype=torch.long)}


class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x)

        # Pass through the fully connected layer
        out = self.fc(hn)
        
        # Apply the ReLU activation function
        out = self.relu(out)
        
        return out
    
class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_lengt, dropout=0 ):
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

        #assert False, (querys.shape, keys.shape, keys.transpose(-2, -1).shape)
        attention_scores = querys @ keys.transpose(-2, -1)

        #attention_scores = querys @ keys.T
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores = attention_scores.masked_fill(mask_bool, -float('inf'))
        
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attention_weights @ values
        


        context_vec = self.dropout(context_vec)
        context_vec = self.out_proj(context_vec)
        return context_vec
    

class DummyGPTmodule(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(GPT_config_124M['vocab_size'], 
                                    GPT_config_124M['emb_dim'])
        self.pos_emb = nn.Embedding(GPT_config_124M['context_length'],
                                    GPT_config_124M['emb_dim'])
        
        self.dropout = nn.Dropout(GPT_config_124M['dropout'])

        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(GPT_config_124M)
                                          for _ in range(GPT_config_124M['n_layers'])])
        
        self.final_norm = DummyLayerNorm(GPT_config_124M['emb_dim'])

        self.out_head = nn.Linear(GPT_config_124M['emb_dim'], GPT_config_124M['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.dropout(x)

        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
                                    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiheadAttention(d_in = cfg['emb_dim'],
                                      d_out = cfg['emb_dim'],
                                      num_heads = cfg['num_heads'],
                                      block_size = cfg['context_length'],
                                      dropout=cfg['dropout'], 
                                      qkv_bias=cfg['qkv_bias'])

        sefl.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg['emb_dim'])
        self.norm2 = nn.LayerNorm(cfg['emb_dim'])

        sefl.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, x): 
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut


        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
    
