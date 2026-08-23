import torch
import torch.nn as nn

class SelfAttentionHead(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        self.Q = nn.Linear(embed_dim, head_dim, bias=False)
        self.K = nn.Linear(embed_dim, head_dim, bias=False)
        self.V = nn.Linear(embed_dim, head_dim, bias=False)
        self.head_dim = head_dim
        

        

    def forward(self, x):
        batch, seq_len, embed_len = x.shape[0], x.shape[1], x.shape[-1]
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        scores = (q @ k.transpose(-2, -1)) / self.head_dim**0.5
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        return attention_weights @ v
        
sample_attention = SelfAttentionHead(16, 4)
test = sample_attention(torch.randn(2, 5, 16))
