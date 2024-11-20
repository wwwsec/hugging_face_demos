from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
transcriber = pipeline(model="openai/whisper-large-v2", device=device, return_timestamps=True)
#result = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
result = transcriber("mlk.mp3")
print(result)