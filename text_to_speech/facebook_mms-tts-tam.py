from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
text_to_speech = pipeline("text-to-speech", model="facebook/mms-tts-tam", device=device)
audio = text_to_speech("Hello, this is a text to speech conversion using Hugging Face models.")
with open("output.wav", "wb") as f:
    f.write(audio["audio"]) 