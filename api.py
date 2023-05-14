import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel
import uvicorn, datetime
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
EXECUTOR_POOL_SIZE = 0
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
MODEL_PATH = "THUDM/chatglm-6b"
LISTEN_ADDR = "0.0.0.0"
LISTEN_PORT = 8000

class Params(BaseModel):
    prompt: str = 'hello'
    history: list[list[str]] = []
    max_length: int = 2048
    top_p: float = 0.7
    temperature: float = 0.95

class Answer(BaseModel):
    status: int = 200
    time: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    response: str
    history: list[list[str]] = []

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/chat")
async def create_chat(params: Params) -> Answer:
    global model, tokenizer
    response, history = model.chat(tokenizer,
                                params.prompt,
                                history=params.history,
                                max_length=params.max_length,
                                top_p=params.top_p,
                                temperature=params.temperature)
    answer_ok = Answer(response=response, history=history)
    # print(answer_ok.json())
    torch_gc()
    return answer_ok

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host=LISTEN_ADDR, port=LISTEN_PORT, workers=1)

