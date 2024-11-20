from transformers import pipeline
import torch
import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = pipeline(model="superb/wav2vec2-base-superb-ks", device=device)
print(classifier.task)