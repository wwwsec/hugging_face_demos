from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
feature_extractor = pipeline("feature-extraction", model="bert-base-uncased", device=device)
result = feature_extractor("Hugging Face is creating a tool that democratizes AI.")
print(result) 