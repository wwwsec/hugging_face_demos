from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
device = 0 if torch.cuda.is_available() else -1

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts",device=device)


embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# print(embeddings_dataset)
speaker_embedding = torch.tensor(embeddings_dataset[20]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])