import torch
from model import GPTModel
from BPE import BPETokenizer
from train import generate

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load("checkpoint.pt", map_location=device)

# 1. vocab 매핑으로 토크나이저를 다시 만들어야 하는데,
#    CharTokenizer는 지금 "텍스트를 받아서" vocab을 만들도록 짜여있죠?
#    저장된 vocab 딕셔너리를 그대로 주입할 방법을 고민해보세요.
#    (힌트: CharTokenizer 인스턴스를 만든 다음, token_ids/token_list를
#     체크포인트에 저장된 걸로 덮어써도 되고, 또는 생성자가 vocab 딕셔너리를
#     직접 받을 수 있게 CharTokenizer를 살짝 고쳐도 됩니다 — 편한 쪽으로)

tok = BPETokenizer(vocab=checkpoint["vocab"], merges=checkpoint["merges"])
seq_len = 16

# 2. config로 모델을 "같은 구조"로 새로 만들기
model = GPTModel(**checkpoint["config"])

# 3. 학습된 가중치 부어넣기
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 4. 생성 테스트 (generate 함수는 train.py에서 복사해오거나 별도 파일로 빼도 됨)
print(generate(model, tok, prompt="ROMEO:", max_new_tokens=500, seq_len=seq_len))

