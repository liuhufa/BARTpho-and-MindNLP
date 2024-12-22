# Author: PAFF
# CreatTime: 2024/11/20
# FileName: predict

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定 GPU ID
#
# # 设置模型名称和目标路径
# model_name = "vinai/bartpho-syllable"
# cache_dir = "./model"
#
# # 下载并加载模型到指定路径
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
#
# print(f"模型和分词器已下载到路径: {cache_dir}")

import mindspore
import os
from mindnlp.transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from tqdm import tqdm  # 导入 tqdm 用于显示进度条

# 定义模型路径
model_name = "vinai/bartpho-syllable"
cache_dir = "./model"
mindspore.set_context(device_target='GPU', device_id=0)
# 加载 MindNLP 模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
model = model
# 数据路径
data_folder = "processed_data"
files = os.listdir(data_folder)

# 初始化 ROUGE 计算器
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

# 遍历处理后的文件并显示进度条
for filename in tqdm(files, desc="Processing files", unit="file"):
    with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f:
        content = f.read()

    # 提取摘要和正文
    parts = content.split("\n\n")
    gold_summary = parts[0].replace("Summary: ", "").strip()
    body_content = parts[1].replace("Body: ", "").strip()

    # 生成摘要
    inputs = tokenizer(body_content, max_length=512, truncation=True, return_tensors="ms")
    summary_ids = model.generate(inputs["input_ids"], max_length=50, num_beams=5, early_stopping=True)
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # 计算 ROUGE 分数
    scores = scorer.score(gold_summary, generated_summary)
    rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
    rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
    rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

# 打印平均 ROUGE 分数
avg_rouge1 = sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"])
avg_rouge2 = sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"])
avg_rougeL = sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"])

print(f"Average ROUGE-1: {avg_rouge1:.4f}, ROUGE-2: {avg_rouge2:.4f}, ROUGE-L: {avg_rougeL:.4f}")

# 保存路径
output_file = "rouge_scores_mindspore.txt"

# 保存结果到文本文件
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Average ROUGE-1: {avg_rouge1:.4f}\n")
    f.write(f"Average ROUGE-2: {avg_rouge2:.4f}\n")
    f.write(f"Average ROUGE-L: {avg_rougeL:.4f}\n")

print(f"ROUGE 分数已保存到 {output_file}")
