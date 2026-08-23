import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model import GPTModel
from BPE import BPETokenizer
from train import generate

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load("checkpoint.pt", weights_only=False, map_location=device)

app = FastAPI()

tok = BPETokenizer(vocab=checkpoint["vocab"], merges=checkpoint["merges"])
model = GPTModel(**checkpoint["config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

SEQ_LEN = checkpoint["config"]["max_seq_len"]

app.mount("/static", StaticFiles(directory="static"), name="static")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    prompt = req.prompt.strip() or "ROMEO:"
    text = generate(
        model,
        tok,
        prompt=prompt,
        max_new_tokens=req.max_new_tokens,
        seq_len=SEQ_LEN,
    )
    return {"text": text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
