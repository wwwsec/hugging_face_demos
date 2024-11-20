# from transformers.pipelines import PIPELINE_REGISTRY

# # 打印所有可用的 pipeline 类型
# print("Available pipelines:")
# for task in PIPELINE_REGISTRY.get_supported_tasks():
#     print(task)

from transformers import pipeline

# 获取所有支持的任务名称
tasks = pipeline.task_to_model.keys()

# 过滤出翻译相关的任务名称
translation_tasks = [task for task in tasks if "translation" in task]

# 打印翻译任务名称
for task in translation_tasks:
    print(task)