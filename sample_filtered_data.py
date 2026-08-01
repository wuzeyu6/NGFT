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

# 第零步：过滤只保留 legal_question_answering 类型
def filter_legal_qa(items):
    filtered = []
    for item in items:
        match = re.match(r'^legal_question_answering', item['id'])
        if match:
            filtered.append(item)
    return filtered

data = filter_legal_qa(data)
print(f"第零步：过滤后剩余 {len(data)} 条（仅 legal_question_answering）")

# 第一步：随机选择5000条
first_sample = random.sample(data, 5000)
print(f"第一步：随机抽取 {len(first_sample)} 条")

# 第二步：筛选output长度大于等于200的数据
filtered_data = []
for item in first_sample:
    if len(item.get('output', '')) >= 200:
        filtered_data.append(item)
print(f"第二步：筛选后剩余 {len(filtered_data)} 条（output长度 >= 200）")

# 第三步：从筛选后的数据中随机选2000条
final_sample = random.sample(filtered_data, min(2000, len(filtered_data)))
print(f"第三步：最终抽取 {len(final_sample)} 条")

# 保存结果
output_file = "/Users/wu/Desktop/llama_factory/DISC-Law-SFT-Alpaca-Test_2000_filtered.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_sample, f, ensure_ascii=False, indent=2)

print(f"\n保存至: {output_file}")

# 统计output长度信息
output_lengths = [len(item['output']) for item in final_sample]
print(f"\n最终数据output长度统计:")
print(f"  最短: {min(output_lengths)}")
print(f"  最长: {max(output_lengths)}")
print(f"  平均: {sum(output_lengths) / len(output_lengths):.1f}")
