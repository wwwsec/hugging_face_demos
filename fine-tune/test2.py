from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("test_trainer/checkpoint-375")

print(tokenizer.tokenize("I love you"))

model = AutoModelForTokenClassification.from_pretrained("test_trainer/checkpoint-375")
print(model)