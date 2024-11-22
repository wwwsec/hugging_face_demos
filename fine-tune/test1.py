from transformers import pipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

classifier = pipeline("sentiment-analysis", model="test_trainer/checkpoint-375", device=device)

print(classifier("I love you"))