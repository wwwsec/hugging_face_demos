from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
model_checkpoint = "facebook/m2m100_418M"
#model_checkpoint = "DDDSSS/translation_en-zh"
translator = pipeline("translation", model=model_checkpoint, device=device)
result = translator("How are you? My name is John.",src_lang="en", tgt_lang="zh")
print(result)