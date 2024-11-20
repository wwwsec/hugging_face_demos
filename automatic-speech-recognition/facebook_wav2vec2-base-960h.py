from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
result = asr("mlk.flac")
print(result) 