from transformers import pipeline
import torch
import os

device = 0 if torch.cuda.is_available() else -1

# Ensure the image path is correct
image_path = "image.png"  # Update this path if necessary

# Check if the file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"The image file {image_path} does not exist.")

# Use a publicly available model
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device=device)
result = image_classifier(image_path)
print(result) 