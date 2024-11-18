from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
model_checkpoint = "Helsinki-NLP/opus-mt-en-zh"
#model_checkpoint = "DDDSSS/translation_en-zh"
translator = pipeline("translation", model=model_checkpoint, device=device)
result = translator("How are you? My name is John.")
print(result)