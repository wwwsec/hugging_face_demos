from transformers import pipeline
import torch
import os
from scipy.io.wavfile import write
import numpy as np

device = 0 if torch.cuda.is_available() else -1
text_to_speech = pipeline("text-to-speech", model="suno/bark-small", device=device)
text = "Hello, this is a text to speech conversion using Hugging Face models."
output = text_to_speech(text)

# Extract audio data and sampling rate
audio = np.array(output["audio"], dtype=np.float32)
sampling_rate = output["sampling_rate"]

# Normalize audio to the range of int16
audio = audio / np.max(np.abs(audio))  # Normalize to -1.0 to 1.0
audio = (audio * 32767).astype(np.int16)  # Convert to int16

# Ensure the output directory exists
if not os.path.exists("output"):
    os.makedirs("output")

# Save the audio to a WAV file
write("suno_bark.wav", sampling_rate, audio)
