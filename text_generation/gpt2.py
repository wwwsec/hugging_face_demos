from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
text_generator = pipeline("text-generation", model="gpt2", device=device)
result = text_generator("Once upon a time", max_length=50, num_return_sequences=1)
print(result) 