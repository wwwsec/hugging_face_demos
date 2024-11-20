from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

# 创建一个英文到中文的翻译管道
en_zh_translator = pipeline("translation_en_to_zh", model="Helsinki-NLP/opus-mt-en-zh", device=device)

# 执行翻译
text = "Hello, how are you?"
translation = en_zh_translator(text)
print(translation) 