import mindspore
import os
from itertools import islice
from mindnlp.transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from tqdm import tqdm

# 定义模型路径
model_name = "vinai/bartpho-syllable"
cache_dir = "./model"

# 设置运行环境（GPU）
mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target='GPU', device_id=0)

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)

# 数据路径
data_folder = "processed_data"
files = os.listdir(data_folder)

# 初始化 ROUGE 计算器
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

# 批量加载函数
def load_batch(file_iter, batch_size):
    """从文件列表中加载批量数据"""
    batch = list(islice(file_iter, batch_size))
    if not batch:
        return None
    data = []
    for filename in batch:
        with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f:
            data.append(f.read())
    return data

# 批量处理文件
batch_size = 8  # 设置批量大小
file_iterator = iter(files)  # 文件迭代器

# 遍历文件并处理
for batch_data in tqdm(iter(lambda: load_batch(file_iterator, batch_size), None), 
                       desc="Processing files", unit="batch"):
    inputs_list = []
    gold_summaries = []

    # 解析每个文件内容
    for content in batch_data:
        parts = content.split("\n\n")
        gold_summaries.append(parts[0].replace("Summary: ", "").strip())
        body_content = parts[1].replace("Body: ", "").strip()
        inputs_list.append(body_content)

    # 批量分词处理
    inputs = tokenizer(inputs_list, max_length=512, truncation=True, padding=True, return_tensors="ms")

    # 批量生成摘要
    summary_ids = model.generate(inputs["input_ids"], max_length=50, num_beams=5, early_stopping=True)
    generated_summaries = [tokenizer.decode(ids, skip_special_tokens=True) for ids in summary_ids]

    # 批量计算 ROUGE 分数
    for gold, generated in zip(gold_summaries, generated_summaries):
        scores = scorer.score(gold, generated)
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

# 计算平均 ROUGE 分数
avg_rouge1 = sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"])
avg_rouge2 = sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"])
avg_rougeL = sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"])

print(f"Average ROUGE-1: {avg_rouge1:.4f}, ROUGE-2: {avg_rouge2:.4f}, ROUGE-L: {avg_rougeL:.4f}")

# 保存结果到文件
output_file = "rouge_scores_mindspore.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Average ROUGE-1: {avg_rouge1:.4f}\n")
    f.write(f"Average ROUGE-2: {avg_rouge2:.4f}\n")
    f.write(f"Average ROUGE-L: {avg_rougeL:.4f}\n")

print(f"ROUGE 分数已保存到 {output_file}")
