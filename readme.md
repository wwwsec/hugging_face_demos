# huggingface 测试

## 创建conda环境
```
conda create -n hf-gpu-env python=3.10
conda activate hf-gpu-env
pip install -r requrement.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

## 模型下载

```
有些模型需要权限，需要使用huggingface-cli login 登录后下载
```

## 查看所有可用的pipelineAvailable pipelines:
audio-classification
automatic-speech-recognition
depth-estimation
document-question-answering
feature-extraction
fill-mask
image-classification
image-feature-extraction
image-segmentation
image-to-image
image-to-text
mask-generation
ner
object-detection
question-answering
sentiment-analysis
summarization
table-question-answering
text-classification
text-generation
text-to-audio
text-to-speech
text2text-generation
token-classification
translation
video-classification
visual-question-answering
vqa
zero-shot-audio-classification
zero-shot-classification
zero-shot-image-classification
zero-shot-object-detection
