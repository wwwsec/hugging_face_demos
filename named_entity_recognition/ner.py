from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
ner_pipeline = pipeline("ner", device=device)
result = ner_pipeline("Hugging Face Inc. is a company based in New York City.")
print(result) 