import json
import random
import re

# 设置随机种子以保证可复现
random.seed(42)

# 加载原始数据
input_file = "/Users/wu/Desktop/llama_factory/DISC-Law-SFT-Alpaca-Test.json"
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"原始数据条数: {len(data)}")

# 过滤只保留 legal_question_answering 类型
def filter_legal_qa(items):
    filtered = []
    for item in items:
        match = re.match(r'^legal_question_answering', item['id'])
        if match:
            filtered.append(item)
    return filtered

filtered_data = filter_legal_qa(data)
print(f"legal_question_answering 类型数据条数: {len(filtered_data)}")

# 从过滤后的数据中随机抽取2000条
sample_size = 2000
if len(filtered_data) < sample_size:
    print(f"警告: 可用数据只有 {len(filtered_data)} 条，少于要求的 {sample_size} 条")
    sampled_data = filtered_data
else:
    sampled_data = random.sample(filtered_data, sample_size)

# 保存抽取的数据
output_file = "/Users/wu/Desktop/llama_factory/DISC-Law-SFT-Alpaca-Test_2000.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print(f"已抽取 {len(sampled_data)} 条 legal_question_answering 类型数据")
print(f"保存至: {output_file}")
