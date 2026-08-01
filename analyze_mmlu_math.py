#!/usr/bin/env python3
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
    """计算单个CSV文件的准确率"""
    import csv
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

def analyze_math_results():
    base_path = "/Users/wu/Desktop/llama_factory/eval_result/mmlu"
    
    results = defaultdict(dict)
    
    # 遍历所有微调类型（GSM, GSM_MASK, DialogSum等）
    for finetune_type in os.listdir(base_path):
        ft_path = os.path.join(base_path, finetune_type)
        if not os.path.isdir(ft_path):
            continue
        
        # 遍历所有模型配置
        for model_config in os.listdir(ft_path):
            model_path = os.path.join(ft_path, model_config)
            if not os.path.isdir(model_path):
                continue
            
            # 读取metrics.json获取总体math准确率
            metrics_path = os.path.join(model_path, "metrics.json")
            math_acc_metrics = None
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    math_acc_metrics = metrics.get('subcat_acc', {}).get('math')
            
            # 计算每个数学科目的准确率
            subject_accs = {}
            for subject in MATH_SUBJECTS:
                csv_path = os.path.join(model_path, f"{subject}.csv")
                acc = calculate_accuracy(csv_path)
                if acc is not None:
                    subject_accs[subject] = acc
            
            if subject_accs or math_acc_metrics is not None:
                key = f"{finetune_type}/{model_config}"
                results[key] = {
                    'metrics_math': math_acc_metrics,
                    'subjects': subject_accs
                }
    
    return results

def print_comparison(results):
    # 分组显示：按模型和微调率
    print("=" * 100)
    print("MMLU 数学科目分析结果")
    print("=" * 100)
    
    # 先按模型类型分组
    model_groups = defaultdict(list)
    for key in results:
        parts = key.split('/')
        ft_type = parts[0]
        model_parts = parts[1].split('_')
        model_name = '_'.join(model_parts[:-1])  # 去掉最后的0.01/0.05/0.10/full
        model_groups[model_name].append((ft_type, key))
    
    for model_name in sorted(model_groups.keys()):
        print(f"\n\n{'=' * 100}")
        print(f"模型: {model_name}")
        print(f"{'=' * 100}")
        
        # 获取该模型的所有配置
        configs = model_groups[model_name]
        
        # 显示表头
        subjects = MATH_SUBJECTS
        print(f"\n{'微调类型/配置':<50}", end="")
        print(f"{'总体Math':<12}", end="")
        for subj in subjects:
            print(f"{subj[:12]:<14}", end="")
        print()
        print("-" * (50 + 12 + 14*len(subjects)))
        
        for ft_type, key in sorted(configs):
            data = results[key]
            config_name = key.split('/')[-1]
            
            print(f"{ft_type}/{config_name:<50}", end="")
            
            if data['metrics_math'] is not None:
                print(f"{data['metrics_math']:<12.2%}", end="")
            else:
                print(f"{'N/A':<12}", end="")
            
            for subj in subjects:
                acc = data['subjects'].get(subj)
                if acc is not None:
                    print(f"{acc:<14.2%}", end="")
                else:
                    print(f"{'N/A':<14}", end="")
            print()

if __name__ == "__main__":
    results = analyze_math_results()
    print_comparison(results)
