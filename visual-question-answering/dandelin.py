from transformers import pipeline
import torch
device = 0 if torch.cuda.is_available() else -1
oracle = pipeline(model="dandelin/vilt-b32-finetuned-vqa", device=device)
print(oracle.task)
image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png"

result = oracle(question="What is she wearing ?", image=image_url)
print(result)

result = oracle(question="What is she wearing ?", image=image_url, top_k=1)
print(result)

result = oracle(question="Is this a person ?", image=image_url, top_k=1)
print(result)

result = oracle(question="Is this a man ?", image=image_url, top_k=1)
print(result)