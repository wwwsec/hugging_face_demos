from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
result = classifier("I love using Hugging Face models!")
print(result) 