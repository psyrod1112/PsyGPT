import torch
import torch.nn as nn
from multihead import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, 4*embed_dim),
                                 nn.GELU(), 
                                 nn.Linear(4*embed_dim, embed_dim))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

if __name__ == "__main__":
    tb = TransformerBlock(16, 4)
    res = tb(torch.randn(2, 5, 16))
    print(res.shape)
