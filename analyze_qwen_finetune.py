#!/usr/bin/env python3
import csv
import json
import os
from collections import defaultdict

# 数学相关的CSV文件
MATH_SUBJECTS = [
    "abstract_algebra",
    "college_mathematics",
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_statistics"
]

def calculate_accuracy(csv_path):
    """计算单个CSV文件的准确率，并返回详细信息"""
    correct = 0
    total = 0
    details = []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)  # skip header
            
            for row in reader:
                total += 1
                is_correct = row[6] == 'True'
                
                if is_correct:
                    correct += 1
                
                details.append({
                    'correct': is_correct,
                    'choiceA': float(row[7]),
                    'choiceB': float(row[8]),
                    'choiceC': float(row[9]),
                    'choiceD': float(row[10])
                })
    except FileNotFoundError:
        return None, None
    
    acc = correct / total if total > 0 else None
    return acc, details

def read_metrics(metrics_path):
    """读取metrics.json"""
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def analyze_finetune_effect():
    base_path = "/Users/wu/Desktop/llama_factory/eval_result/mmlu"
    before_path = os.path.join(base_path, "Qwen2.5-7B-Instruct")
    after_path = os.path.join(base_path, "Qwen2.5-7B-Instruct_full")
    
    print("=" * 120)
    print("Qwen2.5-7B-Instruct 微调GSM8K前后 MMLU 数学表现对比分析")
    print("=" * 120)
    
    # 读取总体指标
    before_metrics = read_metrics(os.path.join(before_path, "metrics.json"))
    after_metrics = read_metrics(os.path.join(after_path, "metrics.json"))
    
    print("\n1. 总体指标对比")
    print("-" * 120)
    
    if before_metrics and after_metrics:
        print(f"{'指标':<30} {'微调前':<15} {'微调后':<15} {'差异':<15}")
        print("-" * 75)
        
        # 总体准确率
        before_acc = before_metrics.get('average_acc')
        after_acc = after_metrics.get('average_acc')
        if before_acc is not None and after_acc is not None:
            diff = after_acc - before_acc
            diff_str = f"{diff:+.1%}"
            print(f"{'总体准确率':<30} {before_acc:>12.1%}{'':<3} {after_acc:>12.1%}{'':<3} {diff_str:>15}")
        
        # Math子分类
        before_math = before_metrics.get('subcat_acc', {}).get('math')
        after_math = after_metrics.get('subcat_acc', {}).get('math')
        if before_math is not None and after_math is not None:
            diff = after_math - before_math
            diff_str = f"{diff:+.1%}"
            print(f"{'Math子分类准确率':<30} {before_math:>12.1%}{'':<3} {after_math:>12.1%}{'':<3} {diff_str:>15}")
        
        # 其他子分类也看看
        print("\n   其他子分类:")
        print(f"   {'子分类':<25} {'微调前':<12} {'微调后':<12} {'差异':<12}")
        print("   " + "-" * 60)
        for subcat in sorted(before_metrics.get('subcat_acc', {}).keys()):
            before = before_metrics.get('subcat_acc', {}).get(subcat)
            after = after_metrics.get('subcat_acc', {}).get(subcat)
            if before is not None and after is not None:
                diff = after - before
                sign = '+' if diff > 0 else ''
                diff_str = f"{sign}{diff:+.1%}"
                if diff < 0:
                    diff_str = f"\033[91m{diff_str}\033[0m"
                elif diff > 0:
                    diff_str = f"\033[92m{diff_str}\033[0m"
                print(f"   {subcat:<25} {before:>10.1%}{'':<2} {after:>10.1%}{'':<2} {diff_str:>12}")
    
    # 各数学科目详细对比
    print("\n\n2. 数学各科目详细对比")
    print("-" * 120)
    
    subject_results = {}
    all_probs_before = defaultdict(list)
    all_probs_after = defaultdict(list)
    
    for subject in MATH_SUBJECTS:
        before_csv = os.path.join(before_path, f"{subject}.csv")
        after_csv = os.path.join(after_path, f"{subject}.csv")
        
        before_acc, before_details = calculate_accuracy(before_csv)
        after_acc, after_details = calculate_accuracy(after_csv)
        
        subject_results[subject] = {
            'before': before_acc,
            'after': after_acc,
            'before_details': before_details,
            'after_details': after_details
        }
        
        # 收集选项概率
        if before_details:
            for d in before_details:
                all_probs_before['A'].append(d['choiceA'])
                all_probs_before['B'].append(d['choiceB'])
                all_probs_before['C'].append(d['choiceC'])
                all_probs_before['D'].append(d['choiceD'])
        
        if after_details:
            for d in after_details:
                all_probs_after['A'].append(d['choiceA'])
                all_probs_after['B'].append(d['choiceB'])
                all_probs_after['C'].append(d['choiceC'])
                all_probs_after['D'].append(d['choiceD'])
    
    print(f"{'科目':<25} {'微调前':<12} {'微调后':<12} {'差异':<12}")
    print("-" * 60)
    
    for subject in MATH_SUBJECTS:
        before = subject_results[subject]['before']
        after = subject_results[subject]['after']
        
        if before is not None and after is not None:
            diff = after - before
            sign = '+' if diff > 0 else ''
            diff_str = f"{sign}{diff:+.1%}"
            if diff < 0:
                diff_str = f"\033[91m{diff_str}\033[0m"
            elif diff > 0:
                diff_str = f"\033[92m{diff_str}\033[0m"
            print(f"{subject:<25} {before:>10.1%}{'':<2} {after:>10.1%}{'':<2} {diff_str:>12}")
    
    # 选项偏好分析
    print("\n\n3. 选项偏好变化分析")
    print("-" * 120)
    print(f"{'选项':<10} {'微调前平均概率':<20} {'微调后平均概率':<20} {'变化':<15}")
    print("-" * 65)
    
    for option in ['A', 'B', 'C', 'D']:
        before_probs = all_probs_before.get(option, [])
        after_probs = all_probs_after.get(option, [])
        
        if before_probs and after_probs:
            before_avg = sum(before_probs) / len(before_probs)
            after_avg = sum(after_probs) / len(after_probs)
            diff = after_avg - before_avg
            sign = '+' if diff > 0 else ''
            diff_str = f"{sign}{diff:+.3f} ({sign}{diff*100:+.1f}%)"
            if diff > 0:
                diff_str = f"\033[92m{diff_str}\033[0m"
            elif diff < 0:
                diff_str = f"\033[91m{diff_str}\033[0m"
            print(f"{option:<10} {before_avg:>15.3f} ({before_avg*100:>5.1f}%){'':<3} {after_avg:>15.3f} ({after_avg*100:>5.1f}%){'':<3} {diff_str:>15}")
    
    # 错误题目对比分析
    print("\n\n4. 关键发现与分析")
    print("=" * 120)
    
    # 分析Micro和Macro变化
    math_decrease = after_math - before_math if (before_math and after_math) else None
    overall_decrease = after_acc - before_acc if (before_acc and after_acc) else None
    
    print("\n主要观察:")
    if math_decrease is not None and math_decrease < 0:
        print(f"• 📉 Math子分类准确率下降: {math_decrease:+.1%}")
    elif math_decrease is not None and math_decrease > 0:
        print(f"• 📈 Math子分类准确率提升: {math_decrease:+.1%}")
    
    if overall_decrease is not None and overall_decrease < 0:
        print(f"• 📉 总体准确率下降: {overall_decrease:+.1%}")
    elif overall_decrease is not None and overall_decrease > 0:
        print(f"• 📈 总体准确率提升: {overall_decrease:+.1%}")
    
    # 分析哪些科目下降最多
    print("\n科目表现变化:")
    decreased = []
    increased = []
    for subject in MATH_SUBJECTS:
        before = subject_results[subject]['before']
        after = subject_results[subject]['after']
        if before is not None and after is not None:
            diff = after - before
            if diff < 0:
                decreased.append((subject, diff))
            elif diff > 0:
                increased.append((subject, diff))
    
    if decreased:
        print("  下降的科目:")
        for subject, diff in sorted(decreased, key=lambda x: x[1]):
            print(f"    - {subject}: {diff:+.1%}")
    
    if increased:
        print("  提升的科目:")
        for subject, diff in sorted(increased, key=lambda x: -x[1]):
            print(f"    - {subject}: {diff:+.1%}")
    
    print("\n\n5. 可能的原因分析")
    print("=" * 120)
    print("""
微调GSM8K后出现能力减弱的可能原因:

1. 灾难性遗忘 (Catastrophic Forgetting)
   • 微调GSM8K数学题时，模型可能过度专注于特定类型的数学题
   • 导致遗忘了原有的通用知识和推理能力
   • MMLU涵盖更广泛的数学领域，与GSM8K题型有差异

2. 分布偏移 (Distribution Shift)
   • GSM8K主要是小学/初中水平的算术应用题
   • MMLU数学涵盖抽象代数、统计学、大学数学等更广泛领域
   • 模型在微调后对新分布过拟合，泛化能力下降

3. 思维链的副作用
   • GSM8K训练时使用了详细的思考过程
   • 但MMLU是直接选择答案，不需要中间推理
   • 可能导致模型在需要直接判断时反而表现下降

4. 答案位置偏好变化
   • 从选项概率变化看，模型可能在微调后产生了特定的答案位置偏好
   • 这种位置偏见可能不适应MMLU的正确答案分布

5. 微调策略问题
   • 可能学习率过大或微调步数过多导致过拟合
   • 没有使用LoRA等参数高效微调方法保留原有能力
   • 缺少混合训练（同时保留原数据和微调数据）
""")

if __name__ == "__main__":
    analyze_finetune_effect()
