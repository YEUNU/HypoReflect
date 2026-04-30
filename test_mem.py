import torch
from sentence_transformers import SentenceTransformer
import os

model_name = "voyageai/voyage-4-nano"
print(f"Loading {model_name}...")
model = SentenceTransformer(model_name, trust_remote_code=True, device="cuda")
print(f"Model loaded.")
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
