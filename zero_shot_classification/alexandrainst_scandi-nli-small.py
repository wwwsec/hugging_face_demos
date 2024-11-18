from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

pipe = pipeline("zero-shot-classification", model="alexandrainst/scandi-nli-small", device=device)

result = pipe("I have a problem with my iphone that needs to be resolved asap!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
)

print(result)