from huggingface_hub import HfApi

# 创建HfApi实例
api = HfApi()

# 搜索英文到中文的翻译模型
#models = api.list_models(filter="audio-classification", search="en-zh")
models = api.list_models(filter="audio-classification")

# 打印模型名称
for model in models:
    print(model.modelId)