from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
text = "Hugging Face is creating a tool that democratizes AI. It is used by many developers around the world."
result = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(result) 