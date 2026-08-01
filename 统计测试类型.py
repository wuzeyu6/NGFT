import json

with open('/Users/wu/Desktop/llama_factory/DISC-Law-SFT-Alpaca-Test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

type_set = set()
type_counts = {}

for item in data:
    id_str = item['id']
    # 提取大类类型：先按下划线分割，再按连字符分割，取前面非数字的部分
    # 找到第一个数字或连字符出现的位置
    import re
    # 使用正则提取类型前缀（所有非数字、非连字符的开头部分）
    match = re.match(r'^([a-zA-Z_]+)', id_str)
    if match:
        type_name = match.group(1).rstrip('_')
    else:
        type_name = id_str
    
    type_set.add(type_name)
    type_counts[type_name] = type_counts.get(type_name, 0) + 1

print("=" * 50)
print("测试集类型统计")
print("=" * 50)
print(f"总数据条数: {len(data)}")
print(f"不同类型数量: {len(type_set)}")
print("\n各类型详情:")
for type_name, count in sorted(type_counts.items()):
    print(f"  - {type_name}: {count} 条")
