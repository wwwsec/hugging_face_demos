from transformers import pipeline
import torch
device = 0 if torch.cuda.is_available() else -1
token_classifier = pipeline(model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple", device=device)
sentence = "Je m'appelle jean-baptiste et je vis à montréal"
tokens = token_classifier(sentence)
print(tokens)