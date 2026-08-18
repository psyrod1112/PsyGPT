import torch
import torch.nn as nn
from attention import SelfAttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, head_dim) for _ in range(num_heads)
        ])
        self.projection = nn.Linear(embed_dim, embed_dim)
            

    def forward(self, x):
        outputs = []
        for head in self.heads:
            output = head(x)
            outputs.append(output)
        
        res = torch.cat(outputs, dim=-1)
        res = self.projection(res)
        return res
    
mha = MultiHeadAttention(16, 4)
res = mha(torch.randn(2, 5, 16))
