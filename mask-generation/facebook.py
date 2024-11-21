from transformers import pipeline
import torch
import torchvision.ops as ops  # Import torchvision for NMS
from PIL import Image
import numpy as np

device = 0 if torch.cuda.is_available() else -1
generator = pipeline(model="facebook/sam-vit-base", task="mask-generation", device=device)

# Define a simple batched_nms function
def batched_nms(boxes, scores, idxs, iou_threshold):
    return ops.batched_nms(boxes, scores, idxs, iou_threshold)

# Use the generator as before
outputs = generator(
    "http://images.cocodataset.org/val2017/000000039769.jpg",
)

print(outputs)
