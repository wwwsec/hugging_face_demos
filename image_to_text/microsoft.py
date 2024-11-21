from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
captioner = pipeline(model="microsoft/git-base", device=device)
result = captioner("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", max_new_tokens=1000)
print(result)