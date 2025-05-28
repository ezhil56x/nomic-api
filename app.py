from pydantic import BaseModel
from fastapi import FastAPI

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

matryoshka_dim = 768

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=8192)
model = AutoModel.from_pretrained('./nomic', trust_remote_code=True, rotary_scaling_factor=2)
model.eval()

app = FastAPI()

class EmbeddingsRequest(BaseModel):
    text: str
    task_type: str

@app.post("/api/embeddings")
async def embeddings(request: EmbeddingsRequest):
    text = request.text
    task_type = request.task_type

    text = f"{task_type}: {text}"

    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = embeddings[:, :matryoshka_dim]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return {"embedding": embeddings[0].tolist()}