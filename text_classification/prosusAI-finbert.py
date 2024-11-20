from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-classification", model="ProsusAI/finbert", device=device)
result = pipe("This restaurant is awesome")
print(result)