import torch
import torch.nn as nn

from transformer_block import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, idx):
        seq_len = idx.shape[1]
        pos_idx = torch.arange(seq_len, device=idx.device)
        token_emb = self.token_embed(idx)
        pos_emb = self.pos_embed(pos_idx)
        embed = token_emb + pos_emb
        for block in self.blocks:
            embed = block(embed)
        fin_embed = self.final_ln(embed)
        logits = self.head(fin_embed)
        
        return logits  # [batch, seq_len, vocab_size]
    
    
if __name__ == "__main__":
    gptModel = GPTModel(vocab_size=8, embed_dim=16, num_heads=4, num_layers=2, max_seq_len=10)
    logits = gptModel(torch.randint(0, 8, (2, 5)))
    print(logits)