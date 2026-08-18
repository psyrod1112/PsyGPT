import torch
import torch.nn.functional as F
from tokenizer import CharTokenizer
from model import GPTModel

# 1. 데이터 준비
text = open("shakespeare.txt").read()
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)

seq_len = 16
batch_size = 8

def get_batch():
    # data에서 랜덤한 시작 위치를 batch_size개 뽑아서
    # x = data[start : start+seq_len], y = data[start+1 : start+seq_len+1] 로 만드세요
    start_ids = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x_list = [] 
    y_list = []
    for start in start_ids:
        x_list.append(data[start:start+seq_len])
        y_list.append(data[start+1:start+seq_len+1])
    
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    return x, y  # 각각 [batch_size, seq_len]

# 2. 모델 + 옵티마이저
model = GPTModel(vocab_size=tok.vocab_size, embed_dim=32, num_heads=4, num_layers=2, max_seq_len=seq_len)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. 학습 루프
for step in range(500):
    x, y = get_batch()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, tok.vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"step {step}, loss {loss.item():.4f}")
