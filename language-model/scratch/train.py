import os
import torch
import torch.nn.functional as F
from BPE import BPETokenizer
from model import GPTModel

# 1. 데이터 준비
text = open("shakespeare.txt").read()
tok = BPETokenizer(text, num_merges=1000)
data = torch.tensor(tok.encode(text), dtype=torch.long)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

seq_len = 128
batch_size = 64

def get_batch(split):
    # data에서 랜덤한 시작 위치를 batch_size개 뽑아서
    # x = data[start : start+seq_len], y = data[start+1 : start+seq_len+1] 로 만드세요
    
    if split == "train":
        start_ids = torch.randint(0, len(train_data) - seq_len - 1, (batch_size,))
        x_list = [] 
        y_list = []
        for start in start_ids:
            x_list.append(train_data[start:start+seq_len])
            y_list.append(train_data[start+1:start+seq_len+1])
        
        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)
        return x, y  # 각각 [batch_size, seq_len]
    
    elif split == "val":
        start_ids = torch.randint(0, len(val_data) - seq_len - 1, (batch_size,))
        x_list = [] 
        y_list = []
        for start in start_ids:
            x_list.append(val_data[start:start+seq_len])
            y_list.append(val_data[start+1:start+seq_len+1])
        
        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)
        return x, y  # 각각 [batch_size, seq_len]

def generate(model, tokenizer, prompt: str, max_new_tokens: int, seq_len: int):
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)  # [1, prompt_len]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-seq_len:]  # 최근 seq_len개만 자르기
            logits = model(idx_cond)
            last_logits = logits[:,-1,:]  # 마지막 위치만
            probs = torch.softmax(last_logits, dim=-1)  # softmax
            next_id = torch.multinomial(probs, num_samples=1)  # multinomial로 샘플링, shape [1, 1]
            idx = torch.cat([idx, next_id], dim=1)  # idx 뒤에 next_id 이어붙이기 (cat, dim=1)
    return tokenizer.decode(idx[0].tolist())

@torch.no_grad()
def estimate_loss(num_batches=20):
    model.eval()
    result = {}
    for split in ["train", "val"]:
        losses = torch.zeros(num_batches)
        for i in range(num_batches):
            x, y = get_batch(split)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, tok.vocab_size), y.view(-1))
            losses[i] = loss.item()
        result[split] = losses.mean().item()
    model.train()
    return result
            

# 2. 모델 + 옵티마이저
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 device: {device}")

if __name__ == "__main__":
    
    model = GPTModel(vocab_size=tok.vocab_size, embed_dim=256, num_heads=8, num_layers=6, max_seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if os.path.exists("checkpoint.pt"):
        checkpoint = torch.load("checkpoint.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("체크포인트에서 이어서 학습합니다")
    else:
        print("체크포인트 없음 — 처음부터 학습합니다")

    best_val_loss = float('inf')
    
    # 3. 학습 루프
    for step in range(20000):
        x, y = get_batch("train")
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, tok.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss:
                checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),# model.state_dict()
                        "config": {
                            "vocab_size": tok.vocab_size,
                            "embed_dim": 256,
                            "num_heads": 8,
                            "num_layers": 6,
                            "max_seq_len": seq_len,
                        },
                        "vocab": tok.vocab,        # 문자 -> 인덱스 매핑 (CharTokenizer가 갖고 있는 그 딕셔너리)
                        "merges": tok.merges,    
                    }
                torch.save(checkpoint, "checkpoint.pt")
                best_val_loss = losses['val']
                print(f"  -> 새 최고기록 (val_loss={losses['val']:.4f}), 저장함")


    print(generate(model, tok, prompt="ROMEO:", max_new_tokens=500, seq_len=seq_len))
    


