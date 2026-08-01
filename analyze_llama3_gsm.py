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

def read_metrics(metrics_path):
    """读取metrics.json"""
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def analyze_model_comparison():
    base_path = "/Users/wu/Desktop/llama_factory/eval_result/mmlu"
    
    # 要分析的模型
    models = [
        {
            'name': 'Meta-Llama-3-8B-Instruct',
            'base_path': os.path.join(base_path, 'base_model', 'Llama3.1-8B-Instruction'),
            'gsm_path': os.path.join(base_path, 'GSM', 'GSM_Meta-Llama-3-8B-Instruct_full'),
            'gsm_mask_path': os.path.join(base_path, 'GSM_MASK', 'GSM_Meta-Llama-3-8B-Instruct_full')
        },
        {
            'name': 'Meta-Llama-3-3B-Instruct',
            'base_path': os.path.join(base_path, 'base_model', 'Llama3.2-3B-Instruction'),
            'gsm_path': os.path.join(base_path, 'GSM', 'GSM_Meta-Llama-3-3B-Instruct_full'),
            'gsm_mask_path': os.path.join(base_path, 'GSM_MASK', 'GSM_Meta-Llama-3-3B-Instruct_full')
        },
        {
            'name': 'Meta-Llama-3-1B-Instruct',
            'base_path': os.path.join(base_path, 'base_model', 'Llama3.2-1B-Instruction'),
            'gsm_path': os.path.join(base_path, 'GSM', 'GSM_Meta-Llama-3-1B-Instruct_full'),
            'gsm_mask_path': os.path.join(base_path, 'GSM_MASK', 'GSM_Meta-Llama-3-1B-Instruct_full')
        }
    ]
    
    print("=" * 140)
    print("MMLU 数学表现对比分析 - 基准模型 vs GSM8K微调 vs GSM8K_MASK微调")
    print("=" * 140)
    
    for model in models:
        print(f"\n\n{'=' * 140}")
        print(f"模型: {model['name']}")
        print(f"{'=' * 140}")
        
        # 检查路径是否存在
        base_exists = os.path.exists(model['base_path'])
        gsm_exists = os.path.exists(model['gsm_path'])
        gsm_mask_exists = os.path.exists(model['gsm_mask_path'])
        
        if not base_exists:
            print(f"⚠️  基准模型路径不存在: {model['base_path']}")
        if not gsm_exists:
            print(f"⚠️  GSM微调模型路径不存在: {model['gsm_path']}")
        if not gsm_mask_exists:
            print(f"⚠️  GSM_MASK微调模型路径不存在: {model['gsm_mask_path']}")
        
        # 读取指标
        base_metrics = read_metrics(os.path.join(model['base_path'], 'metrics.json')) if base_exists else None
        gsm_metrics = read_metrics(os.path.join(model['gsm_path'], 'metrics.json')) if gsm_exists else None
        gsm_mask_metrics = read_metrics(os.path.join(model['gsm_mask_path'], 'metrics.json')) if gsm_mask_exists else None
        
        print("\n1. 总体指标对比")
        print("-" * 140)
        print(f"{'指标':<30} {'基准模型':<15} {'GSM8K微调':<15} {'GSM_MASK':<15} {'GSMvs基准':<15} {'MASKvs基准':<15}")
        print("-" * 110)
        
        # 总体准确率
        base_acc = base_metrics.get('average_acc') if base_metrics else None
        gsm_acc = gsm_metrics.get('average_acc') if gsm_metrics else None
        gsm_mask_acc = gsm_mask_metrics.get('average_acc') if gsm_mask_metrics else None
        
        if base_acc is not None:
            gsm_diff = f"{gsm_acc - base_acc:+.1%}" if gsm_acc else "N/A"
            mask_diff = f"{gsm_mask_acc - base_acc:+.1%}" if gsm_mask_acc else "N/A"
            print(f"{'总体准确率':<30} {base_acc:>12.1%}{'':<3} {gsm_acc:>12.1%}{'':<3} {gsm_mask_acc:>12.1%}{'':<3} {gsm_diff:>15} {mask_diff:>15}")
        
        # Math子分类
        base_math = base_metrics.get('subcat_acc', {}).get('math') if base_metrics else None
        gsm_math = gsm_metrics.get('subcat_acc', {}).get('math') if gsm_metrics else None
        gsm_mask_math = gsm_mask_metrics.get('subcat_acc', {}).get('math') if gsm_mask_metrics else None
        
        if base_math is not None:
            gsm_diff = f"{gsm_math - base_math:+.1%}" if gsm_math else "N/A"
            mask_diff = f"{gsm_mask_math - base_math:+.1%}" if gsm_mask_math else "N/A"
            # 着色
            if gsm_math and gsm_math < base_math:
                gsm_diff = f"\033[91m{gsm_diff}\033[0m"
            elif gsm_math and gsm_math > base_math:
                gsm_diff = f"\033[92m{gsm_diff}\033[0m"
            if gsm_mask_math and gsm_mask_math < base_math:
                mask_diff = f"\033[91m{mask_diff}\033[0m"
            elif gsm_mask_math and gsm_mask_math > base_math:
                mask_diff = f"\033[92m{mask_diff}\033[0m"
            print(f"{'Math子分类准确率':<30} {base_math:>12.1%}{'':<3} {gsm_math:>12.1%}{'':<3} {gsm_mask_math:>12.1%}{'':<3} {gsm_diff:>15} {mask_diff:>15}")
        
        # 各数学科目详细对比
        print("\n2. 数学各科目详细对比")
        print("-" * 140)
        print(f"{'科目':<25} {'基准':<12} {'GSM':<12} {'GSM_MASK':<12} {'GSMvs基准':<15} {'MASKvs基准':<15}")
        print("-" * 90)
        
        subject_results = {}
        for subject in MATH_SUBJECTS:
            base_csv = os.path.join(model['base_path'], f"{subject}.csv")
            gsm_csv = os.path.join(model['gsm_path'], f"{subject}.csv")
            gsm_mask_csv = os.path.join(model['gsm_mask_path'], f"{subject}.csv")
            
            base_acc = calculate_accuracy(base_csv)
            gsm_acc = calculate_accuracy(gsm_csv)
            gsm_mask_acc = calculate_accuracy(gsm_mask_csv)
            
            subject_results[subject] = {
                'base': base_acc,
                'gsm': gsm_acc,
                'gsm_mask': gsm_mask_acc
            }
            
            if base_acc is not None:
                gsm_diff = f"{gsm_acc - base_acc:+.1%}" if gsm_acc else "N/A"
                mask_diff = f"{gsm_mask_acc - base_acc:+.1%}" if gsm_mask_acc else "N/A"
                # 着色
                if gsm_acc and gsm_acc < base_acc:
                    gsm_diff = f"\033[91m{gsm_diff}\033[0m"
                elif gsm_acc and gsm_acc > base_acc:
                    gsm_diff = f"\033[92m{gsm_diff}\033[0m"
                if gsm_mask_acc and gsm_mask_acc < base_acc:
                    mask_diff = f"\033[91m{mask_diff}\033[0m"
                elif gsm_mask_acc and gsm_mask_acc > base_acc:
                    mask_diff = f"\033[92m{mask_diff}\033[0m"
                print(f"{subject:<25} {base_acc:>10.1%}{'':<2} {gsm_acc:>10.1%}{'':<2} {gsm_mask_acc:>10.1%}{'':<2} {gsm_diff:>15} {mask_diff:>15}")
        
        # 其他子分类影响
        print("\n3. 其他子分类变化")
        print("-" * 140)
        if base_metrics and gsm_metrics:
            print(f"{'子分类':<25} {'基准':<12} {'GSM':<12} {'差异':<12}")
            print("-" * 60)
            for subcat in sorted(base_metrics.get('subcat_acc', {}).keys()):
                base = base_metrics.get('subcat_acc', {}).get(subcat)
                gsm = gsm_metrics.get('subcat_acc', {}).get(subcat)
                if base is not None and gsm is not None:
                    diff = gsm - base
                    sign = '+' if diff > 0 else ''
                    diff_str = f"{sign}{diff:+.1%}"
                    if diff < 0:
                        diff_str = f"\033[91m{diff_str}\033[0m"
                    elif diff > 0:
                        diff_str = f"\033[92m{diff_str}\033[0m"
                    print(f"{subcat:<25} {base:>10.1%}{'':<2} {gsm:>10.1%}{'':<2} {diff_str:>12}")
    
    print("\n\n" + "=" * 140)
    print("总结与原因分析")
    print("=" * 140)
    print("""
关键发现:

1. GSM8K微调 vs GSM_MASK微调对比
   • GSM8K使用了完整的思维链(CoT)训练
   • GSM_MASK使用相同数据但遮蔽了思考过程
   • 通过对比可以分析思维链对模型能力的影响

2. 能力减弱的可能原因:
   • 灾难性遗忘: 微调特定任务导致遗忘通用知识
   • 分布偏移: GSM8K题型与MMLU数学题有显著差异
   • 推理范式不匹配: GSM8K需要逐步推理，MMLU需要直接选择
   • 答案位置偏见: 可能在微调后产生了特定位置偏好

3. GSM vs GSM_MASK的对比意义:
   • 如果GSM表现比GSM_MASK差，说明思维链可能干扰了某些能力
   • 如果GSM_MASK表现更好，说明问题可能出在思维链本身而非数据
   • 这可以指导未来的微调策略设计
""")

if __name__ == "__main__":
    analyze_model_comparison()
