#!/usr/bin/env python3
import csv
import os
from collections import defaultdict, Counter

def analyze_csv_details(csv_path):
    """详细分析CSV文件内容"""
    questions = []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # skip header
            
            for idx, row in enumerate(reader):
                question_data = {
                    'index': idx,
                    'correct': row[6] == 'True',
                    'choiceA': float(row[7]),
                    'choiceB': float(row[8]),
                    'choiceC': float(row[9]),
                    'choiceD': float(row[10]),
                    'correct_answer': row[5],
                    'question_text': row[0],
                    'optionA': row[1],
                    'optionB': row[2],
                    'optionC': row[3],
                    'optionD': row[4],
                }
                questions.append(question_data)
    except FileNotFoundError:
        return None
    
    return questions

def get_predicted_answer(q):
    """获取模型预测的答案"""
    probs = {
        'A': q['choiceA'],
        'B': q['choiceB'],
        'C': q['choiceC'],
        'D': q['choiceD'],
    }
    return max(probs, key=probs.get)

def categorize_question(q):
    """尝试分类题目类型"""
    text = q['question_text'].lower()
    
    categories = []
    
    if any(keyword in text for keyword in ['fraction', 'frac', '/']):
        categories.append('分数')
    if any(keyword in text for keyword in ['percentage', 'percent', '%']):
        categories.append('百分比')
    if any(keyword in text for keyword in ['equation', 'solve', 'variable', 'x =', 'y =']):
        categories.append('方程')
    if any(keyword in text for keyword in ['area', 'perimeter', 'volume', 'geometry', 'triangle', 'circle']):
        categories.append('几何')
    if any(keyword in text for keyword in ['average', 'mean', 'median', 'statistics', 'probability']):
        categories.append('统计概率')
    if any(keyword in text for keyword in ['ratio', 'rate', 'proportion']):
        categories.append('比例')
    if any(keyword in text for keyword in ['integer', 'whole number', 'positive', 'negative']):
        categories.append('整数')
    if any(keyword in text for keyword in ['word problem', 'total', 'together', 'combined', 'each', 'every']):
        categories.append('应用题')
    if any(keyword in text for keyword in ['train', 'speed', 'distance', 'time', 'rate']):
        categories.append('速度/距离')
    if any(keyword in text for keyword in ['money', 'cost', 'price', 'dollar', 'cent', 'currency']):
        categories.append('货币')
    
    if not categories:
        categories.append('其他')
    
    return categories

def analyze_elementary_math():
    base_path = "/Users/wu/Desktop/llama_factory/eval_result/mmlu"
    base_csv = os.path.join(base_path, "base_model", "Mistral-7B-Instruct", "elementary_mathematics.csv")
    gsm_csv = os.path.join(base_path, "GSM", "GSM_Mistral-7B-Instruct_full", "elementary_mathematics.csv")
    
    print("=" * 140)
    print("深入分析：为什么微调GSM8K（初等数学）后，初等数学能力反而下降了？")
    print("=" * 140)
    
    base_questions = analyze_csv_details(base_csv)
    gsm_questions = analyze_csv_details(gsm_csv)
    
    if not base_questions or not gsm_questions:
        print("❌ 找不到数据文件")
        return
    
    print(f"\n📊 总体情况")
    print("-" * 140)
    base_acc = sum(1 for q in base_questions if q['correct']) / len(base_questions)
    gsm_acc = sum(1 for q in gsm_questions if q['correct']) / len(gsm_questions)
    print(f"基准模型准确率: {base_acc:.1%} ({sum(1 for q in base_questions if q['correct'])}/{len(base_questions)})")
    print(f"GSM微调后准确率: {gsm_acc:.1%} ({sum(1 for q in gsm_questions if q['correct'])}/{len(gsm_questions)})")
    print(f"变化: {gsm_acc - base_acc:+.1%}")
    
    # 找出基准对但GSM错的题目
    worsened_questions = []
    category_counter = Counter()
    
    for idx, (base_q, gsm_q) in enumerate(zip(base_questions, gsm_questions)):
        if base_q['correct'] and not gsm_q['correct']:
            categories = categorize_question(base_q)
            worsened_questions.append({
                'index': idx,
                'categories': categories,
                'question': base_q['question_text'],
                'correct_answer': base_q['correct_answer'],
                'base_prediction': get_predicted_answer(base_q),
                'gsm_prediction': get_predicted_answer(gsm_q),
                'base_probs': (base_q['choiceA'], base_q['choiceB'], base_q['choiceC'], base_q['choiceD']),
                'gsm_probs': (gsm_q['choiceA'], gsm_q['choiceB'], gsm_q['choiceC'], gsm_q['choiceD']),
            })
            for cat in categories:
                category_counter[cat] += 1
    
    print(f"\n🔍 详细分析：基准正确但GSM错误的题目 ({len(worsened_questions)}) 道")
    print("-" * 140)
    
    # 分类统计
    print(f"\n📂 题目分类统计")
    print("-" * 140)
    for category, count in category_counter.most_common():
        print(f"{category}: {count} 道题")
    
    # 显示具体题目
    print(f"\n📝 具体题目示例（前10道）")
    print("-" * 140)
    for i, q in enumerate(worsened_questions[:10]):
        print(f"\n题目 {i+1} (原题号 {q['index']+1})")
        print(f"分类: {', '.join(q['categories'])}")
        print(f"题目: {q['question'][:150]}...")
        print(f"正确答案: {q['correct_answer']}")
        print(f"基准模型预测: {q['base_prediction']} (正确)")
        print(f"GSM模型预测: {q['gsm_prediction']} (错误)")
        print(f"基准概率: A={q['base_probs'][0]:.2%}, B={q['base_probs'][1]:.2%}, C={q['base_probs'][2]:.2%}, D={q['base_probs'][3]:.2%}")
        print(f"GSM概率: A={q['gsm_probs'][0]:.2%}, B={q['gsm_probs'][1]:.2%}, C={q['gsm_probs'][2]:.2%}, D={q['gsm_probs'][3]:.2%}")
    
    # 分析预测变化
    print(f"\n🎯 答案预测变化分析")
    print("-" * 140)
    
    prediction_changes = Counter()
    for q in worsened_questions:
        key = f"{q['base_prediction']}→{q['gsm_prediction']}"
        prediction_changes[key] += 1
    
    print("从正确答案变成了什么：")
    for change, count in prediction_changes.most_common():
        print(f"  {change}: {count} 次")
    
    # 找出GSM对但基准错的题目（提升的）
    improved_questions = []
    for idx, (base_q, gsm_q) in enumerate(zip(base_questions, gsm_questions)):
        if not base_q['correct'] and gsm_q['correct']:
            categories = categorize_question(base_q)
            improved_questions.append({
                'index': idx,
                'categories': categories,
            })
    
    print(f"\n✅ 提升的题目：基准错误但GSM正确 ({len(improved_questions)}) 道")
    if improved_questions:
        improve_counter = Counter()
        for q in improved_questions:
            for cat in q['categories']:
                improve_counter[cat] += 1
        print("分类统计：")
        for category, count in improve_counter.most_common():
            print(f"  {category}: {count} 道题")
    
    print("\n\n" + "=" * 140)
    print("原因深度分析")
    print("=" * 140)
    print("""
为什么同样是初等数学，GSM8K微调后能力反而下降？

📊 可能的关键原因：

1. **题型差异巨大**
   • GSM8K: 主要是小学/初中算术应用题，有明确的解题步骤
   • MMLU初等数学: 涵盖更广（代数、几何、统计等），题型更多样
   • 模型对GSM8K的特定题型过拟合，泛化到其他初等数学题时能力下降

2. **推理范式不匹配**
   • GSM8K训练: 需要生成详细的思维链和完整解答
   • MMLU测试: 只需要从4个选项中选择，不需要中间推理
   • 模型在微调后可能"过度思考"，反而在快速选择时表现下降

3. **答案格式和位置偏见**
   • GSM8K的正确答案通常在末尾或有特定格式
   • MMLU的正确答案位置更随机
   • 从之前的分析看，GSM微调后模型对选项A偏好显著上升
   • 这可能是训练数据中答案分布导致的偏见

4. **难度分布不同**
   • GSM8K: 难度相对集中在小学-初中水平
   • MMLU初等数学: 可能包含一些更有挑战性的题目
   • 模型在特定难度范围内过拟合，超出范围时表现下降

5. **灾难性遗忘**
   • 微调时模型可能用GSM8K的模式覆盖了原有的数学推理模式
   • 即使同样是初等数学，原有的某些推理策略被"遗忘"了

6. **思维链的副作用**
   • 思维链训练可能让模型变得"谨慎"，需要更多推理步骤
   • 但在选择题中，有时直觉或快速判断更有效
   • GSM_MASK表现更好也佐证了这一点

💡 这也说明：
   • 微调数据的多样性很重要
   • 不要过度依赖特定的训练范式
   • 有时简单的方法（如GSM_MASK）反而更有效
""")

if __name__ == "__main__":
    analyze_elementary_math()
