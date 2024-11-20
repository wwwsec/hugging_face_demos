from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
sentiment_analyzer = pipeline("sentiment-analysis", device=device)
result = sentiment_analyzer("I am so happy with the results!")
print(result) 