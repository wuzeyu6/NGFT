#!/usr/bin/env python3
import csv
import json
import os
from collections import defaultdict, Counter

def calculate_accuracy(csv_path):
    """计算单个CSV文件的准确率"""
    correct = 0
    total = 0
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                total += 1
                if row[6] == 'True':
                    correct += 1
    except FileNotFoundError:
        return None
    return correct / total if total > 0 else None

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

def read_metrics(metrics_path):
    """读取metrics.json"""
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def analyze_bioinstruct():
    base_path = "/Users/wu/Desktop/llama_factory/eval_result/mmlu"
    base_model_path = os.path.join(base_path, "base_model", "Mistral-7B-Instruct")
    bioinstruct_path = os.path.join(base_path, "BioInstruct", "BioInstruct_Mistral-7B-Instruct_full")
    bioinstruct_mask_path = os.path.join(base_path, "BioInstruct_MASK", "BioInstruct_Mistral-7B-Instruct_full")
    
    print("=" * 140)
    print("BioInstruct微调影响分析 - MMLU生物相关题目")
    print("=" * 140)
    
    # 生物相关的CSV文件
    bio_subjects = [
        "anatomy",
        "college_biology",
        "college_medicine",
        "human_aging",
        "human_sexuality",
        "medical_genetics",
        "nutrition",
        "virology",
    ]
    
    # 读取指标
    base_metrics = read_metrics(os.path.join(base_model_path, "metrics.json"))
    bio_metrics = read_metrics(os.path.join(bioinstruct_path, "metrics.json"))
    bio_mask_metrics = read_metrics(os.path.join(bioinstruct_mask_path, "metrics.json")) if os.path.exists(bioinstruct_mask_path) else None
    
    print("\n1. 总体指标对比")
    print("-" * 140)
    
    if base_metrics and bio_metrics:
        print(f"{'指标':<30} {'基准模型':<15} {'BioInstruct':<15} {'BioInstruct_MASK':<15} {'Bio变化':<15} {'MASK变化':<15}")
        print("-" * 140)
        
        base_acc = base_metrics.get('average_acc')
        bio_acc = bio_metrics.get('average_acc')
        bio_mask_acc = bio_mask_metrics.get('average_acc') if bio_mask_metrics else None
        
        print(f"{'总体准确率':<30} {base_acc:>12.1%}{'':<3} {bio_acc:>12.1%}{'':<3} {bio_mask_acc:>12.1%}{'':<3} {bio_acc - base_acc:>+12.1%} {bio_mask_acc - base_acc:>+12.1%}")
        
        print("\n2. 各生物科目详细表现")
        print("-" * 140)
        print(f"{'科目':<25} {'基准':<12} {'BioInstruct':<12} {'Bio_MASK':<12} {'Bio变化':<15} {'MASK变化':<15}")
        print("-" * 115)
        
        subject_results = {}
        all_probs_base = defaultdict(list)
        all_probs_bio = defaultdict(list)
        all_probs_bio_mask = defaultdict(list)
        
        for subject in bio_subjects:
            base_csv = os.path.join(base_model_path, f"{subject}.csv")
            bio_csv = os.path.join(bioinstruct_path, f"{subject}.csv")
            bio_mask_csv = os.path.join(bioinstruct_mask_path, f"{subject}.csv") if os.path.exists(bioinstruct_mask_path) else None
            
            base_acc = calculate_accuracy(base_csv)
            bio_acc = calculate_accuracy(bio_csv)
            bio_mask_acc = calculate_accuracy(bio_mask_csv) if bio_mask_csv else None
            
            subject_results[subject] = {
                'base': base_acc,
                'bio': bio_acc,
                'bio_mask': bio_mask_acc,
            }
            
            # 收集概率数据
            base_q = analyze_csv_details(base_csv)
            bio_q = analyze_csv_details(bio_csv)
            bio_mask_q = analyze_csv_details(bio_mask_csv) if bio_mask_csv else None
            
            if base_q:
                for q in base_q:
                    all_probs_base['A'].append(q['choiceA'])
                    all_probs_base['B'].append(q['choiceB'])
                    all_probs_base['C'].append(q['choiceC'])
                    all_probs_base['D'].append(q['choiceD'])
            
            if bio_q:
                for q in bio_q:
                    all_probs_bio['A'].append(q['choiceA'])
                    all_probs_bio['B'].append(q['choiceB'])
                    all_probs_bio['C'].append(q['choiceC'])
                    all_probs_bio['D'].append(q['choiceD'])
            
            if bio_mask_q:
                for q in bio_mask_q:
                    all_probs_bio_mask['A'].append(q['choiceA'])
                    all_probs_bio_mask['B'].append(q['choiceB'])
                    all_probs_bio_mask['C'].append(q['choiceC'])
                    all_probs_bio_mask['D'].append(q['choiceD'])
            
            bio_diff = bio_acc - base_acc if bio_acc and base_acc else None
            bio_mask_diff = bio_mask_acc - base_acc if bio_mask_acc and base_acc else None
            
            bio_diff_str = f"{bio_diff:+.1%}" if bio_diff is not None else "N/A"
            bio_mask_diff_str = f"{bio_mask_diff:+.1%}" if bio_mask_diff is not None else "N/A"
            
            print(f"{subject:<25} {base_acc:>10.1%}{'':<2} {bio_acc:>10.1%}{'':<2} {bio_mask_acc:>10.1%}{'':<2} {bio_diff_str:>15} {bio_mask_diff_str:>15}")
        
        # 选项偏好变化
        print(f"\n3. 选项偏好变化分析（生物相关题目）")
        print("-" * 140)
        print(f"{'选项':<10} {'基准平均':<15} {'BioInstruct':<15} {'Bio_MASK':<15} {'Bio变化':<15} {'MASK变化':<15}")
        print("-" * 100)
        
        for option in ['A', 'B', 'C', 'D']:
            base_avg = sum(all_probs_base[option]) / len(all_probs_base[option]) if all_probs_base[option] else None
            bio_avg = sum(all_probs_bio[option]) / len(all_probs_bio[option]) if all_probs_bio[option] else None
            bio_mask_avg = sum(all_probs_bio_mask[option]) / len(all_probs_bio_mask[option]) if all_probs_bio_mask[option] else None
            
            if base_avg is not None:
                bio_diff = bio_avg - base_avg if bio_avg else None
                bio_mask_diff = bio_mask_avg - base_avg if bio_mask_avg else None
                
                bio_diff_str = f"{bio_diff:+.3f} ({bio_diff*100:+.1f}%)" if bio_diff is not None else "N/A"
                bio_mask_diff_str = f"{bio_mask_diff:+.3f} ({bio_mask_diff*100:+.1f}%)" if bio_mask_diff is not None else "N/A"
                
                print(f"{option:<10} {base_avg:>12.3f} ({base_avg*100:>5.1f}%){'':<3} {bio_avg:>12.3f} ({bio_avg*100:>5.1f}%){'':<3} {bio_mask_avg:>12.3f} ({bio_mask_avg*100:>5.1f}%){'':<3} {bio_diff_str:>15} {bio_mask_diff_str:>15}")
        
        # 详细分析college_biology
        print(f"\n4. 深入分析：college_biology")
        print("-" * 140)
        
        base_bio_q = analyze_csv_details(os.path.join(base_model_path, "college_biology.csv"))
        bio_bio_q = analyze_csv_details(os.path.join(bioinstruct_path, "college_biology.csv"))
        
        if base_bio_q and bio_bio_q:
            worsened = []
            improved = []
            
            for idx, (base_q, bio_q) in enumerate(zip(base_bio_q, bio_bio_q)):
                if base_q['correct'] and not bio_q['correct']:
                    worsened.append({
                        'index': idx,
                        'question': base_q['question_text'],
                        'correct_answer': base_q['correct_answer'],
                        'base_pred': get_predicted_answer(base_q),
                        'bio_pred': get_predicted_answer(bio_q),
                    })
                elif not base_q['correct'] and bio_q['correct']:
                    improved.append({
                        'index': idx,
                    })
            
            print(f"基准正确但BioInstruct错误: {len(worsened)} 道题")
            print(f"基准错误但BioInstruct正确: {len(improved)} 道题")
            
            if worsened:
                print(f"\n具体题目示例（前5道）:")
                for i, q in enumerate(worsened[:5]):
                    print(f"\n题目 {i+1}: {q['question'][:100]}...")
                    print(f"正确答案: {q['correct_answer']}, 基准预测: {q['base_pred']}, Bio预测: {q['bio_pred']}")
                
                # 分析答案变化
                prediction_changes = Counter()
                for q in worsened:
                    key = f"{q['base_pred']}→{q['bio_pred']}"
                    prediction_changes[key] += 1
                
                print(f"\n答案变化统计:")
                for change, count in prediction_changes.most_common():
                    print(f"  {change}: {count} 次")
    
    print("\n\n" + "=" * 140)
    print("关键发现与原因分析")
    print("=" * 140)
    print("""
📊 可能的原因：

1. 灾难性遗忘
   • 微调BioInstruct后，模型可能过度专注于生物医学特定内容
   • 遗忘了原有的通用知识和推理能力

2. 题型差异
   • BioInstruct和MMLU生物题可能在题型、难度分布上有差异
   • 模型对BioInstruct的特定题型过拟合

3. 答案位置偏见
   • 类似GSM8K的发现，BioInstruct训练数据可能有特定的答案位置分布
   • 模型学到了位置偏见而非真正的生物知识

4. 思维链的副作用
   • 如果BioInstruct用了思维链训练，可能让模型过度复杂化简单问题
   • 对比BioInstruct_MASK可以验证这一点

5. 数据领域差异
   • BioInstruct可能侧重某些生物医学子领域
   • MMLU的生物题目范围更广，导致模型泛化能力下降
""")

if __name__ == "__main__":
    analyze_bioinstruct()
