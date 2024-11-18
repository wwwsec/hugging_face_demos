from transformers import pipeline
import torch
import os

# 检查是否有可用的GPU
device = 0 if torch.cuda.is_available() else -1

# 创建一个文本到语音的pipeline
text_to_speech = pipeline("text-to-speech", model="suno/bark", device=device)

# 输入文本
text = "Hello, this is a text to speech conversion using Hugging Face models."

# 生成语音
audio = text_to_speech(text)

# 创建输出目录
if not os.path.exists("output"):
    os.makedirs("output")

# 保存音频文件    
with open("output/suno_bark.wav", "wb") as f:
    f.write(audio["audio"])
