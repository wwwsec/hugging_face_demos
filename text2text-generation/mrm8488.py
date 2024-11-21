from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
generator = pipeline(model="mrm8488/t5-base-finetuned-question-generation-ap", device=device, max_new_tokens=200)
result = generator(
    "answer: Manuel context: Manuel has created RuPERTa-base with the support of HF-Transformers and Google"
)
print(result)