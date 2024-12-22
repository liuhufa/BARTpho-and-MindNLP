# Author: PAFF
# CreatTime: 2024/11/20
# FileName: data_pre
import os

# 输入文件夹路径
input_folder = "test_tokenized"
output_folder = "processed_data"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有文件
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按段落分割文本
    paragraphs = content.split("\n\n")
    if len(paragraphs) < 2:
        print(f"文件 {filename} 内容不足，跳过处理。")
        continue

    # 提取摘要和正文
    gold_summary = paragraphs[0].strip()
    body_content = "\n\n".join(paragraphs[1:]).strip()

    # 保存处理后的数据
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Summary: {gold_summary}\n\nBody: {body_content}")
