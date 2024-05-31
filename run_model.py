import torch
import torch.nn as nn

inputs = torch.tensor(
[[0.43, 0.15, 0.89], 
[0.55, 0.87, 0.66], 
[0.57, 0.85, 0.64], 
[0.22, 0.58, 0.33],
[0.77, 0.25, 0.10], 
[0.05, 0.80, 0.55]] 
)


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        #self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_keys = nn.Parameter(torch.rand(d_in, d_out))
        self.W_values = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_keys 
        querys = x @ self.W_query
        values = x @ self.W_values

        attention_scores = querys @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1)
        
        context_vec = attention_weights @ values
        return context_vec
    
torch.manual_seed(123)

d_in = inputs.shape[1]
d_out = 2

#safsdaf
sa_v1 = SelfAttention(3, 2)
sa_v1(inputs)
