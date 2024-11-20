from transformers import pipeline
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = pipeline(model="superb/wav2vec2-base-superb-ks", device=device)
result = classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
print(result)