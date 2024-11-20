from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
fill_mask = pipeline("fill-mask", model="bert-base-uncased", device=device)
result = fill_mask("Hugging Face is creating a [MASK] that democratizes AI.")
print(result) 
