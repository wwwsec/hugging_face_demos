import torch
import soundfile as sf
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1
synthesiser = pipeline("text-to-audio", "facebook/musicgen-stereo-small", device=device, torch_dtype=torch.float16)

music = synthesiser("lo-fi music with a soothing melody", forward_params={"max_new_tokens": 256})

sf.write("facebookmusicgen_out.wav", music["audio"][0].T, music["sampling_rate"])