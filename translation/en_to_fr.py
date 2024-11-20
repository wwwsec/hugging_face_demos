from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
en_fr_translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr", device=device)
result = en_fr_translator("How old are you?")
print(result)