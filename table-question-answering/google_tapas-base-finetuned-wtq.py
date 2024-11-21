from transformers import pipeline
import torch
device = 0 if torch.cuda.is_available() else -1
oracle = pipeline(model="google/tapas-base-finetuned-wtq", device=device)
table = {
    "Repository": ["Transformers", "Datasets", "Tokenizers"],
    "Stars": ["36542", "4512", "3934"],
    "Contributors": ["651", "77", "34"],
    "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
}   
result = oracle(query="How many stars does the transformers repository have?", table=table)
print(result)