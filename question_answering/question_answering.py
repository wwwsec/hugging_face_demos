from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
qa_pipeline = pipeline(model="deepset/roberta-base-squad2", device=device)
# qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
context = "Hugging Face is creating a tool that democratizes AI."
question = "What is Hugging Face creating?"
result = qa_pipeline(question=question, context=context)
print(result) 