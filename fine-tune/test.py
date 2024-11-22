from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the fine-tuned model and tokenizer
model_name = "test_trainer/checkpoint-375"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare the input text
text = "This is an example text for inference."
inputs = tokenizer(text, return_tensors="pt")

# Run the model
with torch.no_grad():
    outputs = model(**inputs)

# Process the output
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()

# Print the predicted class
print(f"Predicted class: {predicted_class}")