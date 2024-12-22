# Author: PAFF
# CreatTime: 2024/11/20
# FileName: predict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from tqdm import tqdm  # 导入 tqdm 用于显示进度条
import torch
import os

# 模型名称和缓存路径
model_name = "vinai/bartpho-syllable"
cache_dir = "./model"

# 检查 GPU 是否可用
if not torch.cuda.is_available():
    raise EnvironmentError("当前环境未检测到 GPU，请检查 CUDA 驱动或使用 CPU 运行。")

# 下载并加载模型到指定路径
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)

# 使用 DataParallel 实现多 GPU 并行
model = torch.nn.DataParallel(model). cuda()

# 数据路径
data_folder = "processed_data"
files = os.listdir(data_folder)

# 初始化 ROUGE 计算器
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

# 遍历处理后的文件并显示进度条
for filename in tqdm(files, desc="Processing files", unit="file"):
    # 读取文件内容
    with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f:
        content = f.read()

    # 提取摘要和正文
    parts = content.split("\n\n")
    if len(parts) < 2:
        print(f"文件 {filename} 内容格式有误，跳过。")
        continue
    gold_summary = parts[0].replace("Summary: ", "").strip()
    body_content = parts[1].replace("Body: ", "").strip()

    # 生成摘要（将输入移动到 GPU）
    inputs = tokenizer(body_content, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    summary_ids = model.module.generate(inputs["input_ids"], max_length=50, num_beams=5, early_stopping=True)
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # 计算 ROUGE 分数
    scores = scorer.score(gold_summary, generated_summary)
    rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
    rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
    rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

# 计算平均 ROUGE 分数
avg_rouge1 = sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"])
avg_rouge2 = sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"])
avg_rougeL = sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"])

# 打印结果
print(f"Average ROUGE-1: {avg_rouge1:.4f}, ROUGE-2: {avg_rouge2:.4f}, ROUGE-L: {avg_rougeL:.4f}")

# 保存路径
output_file = "rouge_scores.txt"  # 或 "rouge_scores.csv"

# 保存结果到文本文件
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Average ROUGE-1: {avg_rouge1:.4f}\n")
    f.write(f"Average ROUGE-2: {avg_rouge2:.4f}\n")
    f.write(f"Average ROUGE-L: {avg_rougeL:.4f}\n")

    # # 如果需要保存详细的每个文件的分数：
    # f.write("\nDetailed ROUGE Scores:\n")
    # f.write("File,ROUGE-1,ROUGE-2,ROUGE-L\n")
    # for i, filename in enumerate(files):
    #     f.write(f"{filename},{rouge_scores['rouge1'][i]:.4f},{rouge_scores['rouge2'][i]:.4f},{rouge_scores['rougeL'][i]:.4f}\n")

print(f"ROUGE 分数已保存到 {output_file}")