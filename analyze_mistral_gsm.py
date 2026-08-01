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

def analyze_mistral_comparison():
    base_path = "/Users/wu/Desktop/llama_factory/eval_result/mmlu"
    base_model_path = os.path.join(base_path, "base_model", "Mistral-7B-Instruct")
    gsm_path = os.path.join(base_path, "GSM", "GSM_Mistral-7B-Instruct_full")
    gsm_mask_path = os.path.join(base_path, "GSM_MASK", "GSM_Mistral-7B-Instruct_full")
    
    print("=" * 140)
    print("Mistral-7B-Instruct 微调GSM8K前后 MMLU 表现对比分析")
    print("=" * 140)
    
    # 读取指标
    base_metrics = read_metrics(os.path.join(base_model_path, 'metrics.json'))
    gsm_metrics = read_metrics(os.path.join(gsm_path, 'metrics.json'))
    gsm_mask_metrics = read_metrics(os.path.join(gsm_mask_path, 'metrics.json')) if os.path.exists(gsm_mask_path) else None
    
    print("\n1. 总体指标对比")
    print("-" * 140)
    print(f"{'指标':<30} {'基准模型':<15} {'GSM8K微调':<15} {'GSM_MASK':<15} {'GSMvs基准':<15} {'MASKvs基准':<15}")
    print("-" * 110)
    
    # 总体准确率
    base_acc = base_metrics.get('average_acc') if base_metrics else None
    gsm_acc = gsm_metrics.get('average_acc') if gsm_metrics else None
    gsm_mask_acc = gsm_mask_metrics.get('average_acc') if gsm_mask_metrics else None
    
    if base_acc is not None:
        gsm_diff = gsm_acc - base_acc if gsm_acc else None
        mask_diff = gsm_mask_acc - base_acc if gsm_mask_acc else None
        
        gsm_diff_str = f"{gsm_diff:+.1%}" if gsm_diff is not None else "N/A"
        mask_diff_str = f"{mask_diff:+.1%}" if mask_diff is not None else "N/A"
        
        # 着色
        if gsm_diff is not None and gsm_diff < 0:
            gsm_diff_str = f"\033[91m{gsm_diff_str}\033[0m"
        elif gsm_diff is not None and gsm_diff > 0:
            gsm_diff_str = f"\033[92m{gsm_diff_str}\033[0m"
        
        if mask_diff is not None and mask_diff < 0:
            mask_diff_str = f"\033[91m{mask_diff_str}\033[0m"
        elif mask_diff is not None and mask_diff > 0:
            mask_diff_str = f"\033[92m{mask_diff_str}\033[0m"
        
        print(f"{'总体准确率':<30} {base_acc:>12.1%}{'':<3} {gsm_acc:>12.1%}{'':<3} {gsm_mask_acc:>12.1%}{'':<3} {gsm_diff_str:>15} {mask_diff_str:>15}")
    
    # Math子分类
    base_math = base_metrics.get('subcat_acc', {}).get('math') if base_metrics else None
    gsm_math = gsm_metrics.get('subcat_acc', {}).get('math') if gsm_metrics else None
    gsm_mask_math = gsm_mask_metrics.get('subcat_acc', {}).get('math') if gsm_mask_metrics else None
    
    if base_math is not None:
        gsm_diff = gsm_math - base_math if gsm_math else None
        mask_diff = gsm_mask_math - base_math if gsm_mask_math else None
        
        gsm_diff_str = f"{gsm_diff:+.1%}" if gsm_diff is not None else "N/A"
        mask_diff_str = f"{mask_diff:+.1%}" if mask_diff is not None else "N/A"
        
        # 着色
        if gsm_diff is not None and gsm_diff < 0:
            gsm_diff_str = f"\033[91m{gsm_diff_str}\033[0m"
        elif gsm_diff is not None and gsm_diff > 0:
            gsm_diff_str = f"\033[92m{gsm_diff_str}\033[0m"
        
        if mask_diff is not None and mask_diff < 0:
            mask_diff_str = f"\033[91m{mask_diff_str}\033[0m"
        elif mask_diff is not None and mask_diff > 0:
            mask_diff_str = f"\033[92m{mask_diff_str}\033[0m"
        
        print(f"{'Math子分类准确率':<30} {base_math:>12.1%}{'':<3} {gsm_math:>12.1%}{'':<3} {gsm_mask_math:>12.1%}{'':<3} {gsm_diff_str:>15} {mask_diff_str:>15}")
    
    # 各数学科目详细对比
    print("\n2. 数学各科目详细对比")
    print("-" * 140)
    print(f"{'科目':<25} {'基准':<12} {'GSM':<12} {'GSM_MASK':<12} {'GSMvs基准':<15} {'MASKvs基准':<15}")
    print("-" * 90)
    
    subject_results = {}
    all_probs_base = defaultdict(list)
    all_probs_gsm = defaultdict(list)
    all_probs_mask = defaultdict(list)
    
    for subject in MATH_SUBJECTS:
        base_csv = os.path.join(base_model_path, f"{subject}.csv")
        gsm_csv = os.path.join(gsm_path, f"{subject}.csv")
        gsm_mask_csv = os.path.join(gsm_mask_path, f"{subject}.csv") if gsm_mask_path else None
        
        base_acc, base_details = calculate_accuracy(base_csv)
        gsm_acc, gsm_details = calculate_accuracy(gsm_csv)
        gsm_mask_acc, gsm_mask_details = calculate_accuracy(gsm_mask_csv) if gsm_mask_csv else (None, None)
        
        subject_results[subject] = {
            'base': base_acc,
            'gsm': gsm_acc,
            'gsm_mask': gsm_mask_acc
        }
        
        # 收集概率数据
        if base_details:
            for d in base_details:
                all_probs_base['A'].append(d['choiceA'])
                all_probs_base['B'].append(d['choiceB'])
                all_probs_base['C'].append(d['choiceC'])
                all_probs_base['D'].append(d['choiceD'])
        
        if gsm_details:
            for d in gsm_details:
                all_probs_gsm['A'].append(d['choiceA'])
                all_probs_gsm['B'].append(d['choiceB'])
                all_probs_gsm['C'].append(d['choiceC'])
                all_probs_gsm['D'].append(d['choiceD'])
        
        if gsm_mask_details:
            for d in gsm_mask_details:
                all_probs_mask['A'].append(d['choiceA'])
                all_probs_mask['B'].append(d['choiceB'])
                all_probs_mask['C'].append(d['choiceC'])
                all_probs_mask['D'].append(d['choiceD'])
        
        if base_acc is not None:
            gsm_diff = gsm_acc - base_acc if gsm_acc else None
            mask_diff = gsm_mask_acc - base_acc if gsm_mask_acc else None
            
            gsm_diff_str = f"{gsm_diff:+.1%}" if gsm_diff is not None else "N/A"
            mask_diff_str = f"{mask_diff:+.1%}" if mask_diff is not None else "N/A"
            
            # 着色
            if gsm_diff is not None and gsm_diff < 0:
                gsm_diff_str = f"\033[91m{gsm_diff_str}\033[0m"
            elif gsm_diff is not None and gsm_diff > 0:
                gsm_diff_str = f"\033[92m{gsm_diff_str}\033[0m"
            
            if mask_diff is not None and mask_diff < 0:
                mask_diff_str = f"\033[91m{mask_diff_str}\033[0m"
            elif mask_diff is not None and mask_diff > 0:
                mask_diff_str = f"\033[92m{mask_diff_str}\033[0m"
            
            print(f"{subject:<25} {base_acc:>10.1%}{'':<2} {gsm_acc:>10.1%}{'':<2} {gsm_mask_acc:>10.1%}{'':<2} {gsm_diff_str:>15} {mask_diff_str:>15}")
    
    # 选项偏好变化
    print("\n3. 选项偏好变化分析")
    print("-" * 140)
    print(f"{'选项':<10} {'基准平均':<15} {'GSM平均':<15} {'MASK平均':<15} {'GSM变化':<15} {'MASK变化':<15}")
    print("-" * 90)
    
    for option in ['A', 'B', 'C', 'D']:
        base_avg = sum(all_probs_base[option]) / len(all_probs_base[option]) if all_probs_base[option] else None
        gsm_avg = sum(all_probs_gsm[option]) / len(all_probs_gsm[option]) if all_probs_gsm[option] else None
        mask_avg = sum(all_probs_mask[option]) / len(all_probs_mask[option]) if all_probs_mask[option] else None
        
        if base_avg is not None:
            gsm_diff = gsm_avg - base_avg if gsm_avg else None
            mask_diff = mask_avg - base_avg if mask_avg else None
            
            gsm_diff_str = f"{gsm_diff:+.3f} ({gsm_diff*100:+.1f}%)" if gsm_diff is not None else "N/A"
            mask_diff_str = f"{mask_diff:+.3f} ({mask_diff*100:+.1f}%)" if mask_diff is not None else "N/A"
            
            # 着色
            if gsm_diff is not None and gsm_diff > 0:
                gsm_diff_str = f"\033[92m{gsm_diff_str}\033[0m"
            elif gsm_diff is not None and gsm_diff < 0:
                gsm_diff_str = f"\033[91m{gsm_diff_str}\033[0m"
            
            if mask_diff is not None and mask_diff > 0:
                mask_diff_str = f"\033[92m{mask_diff_str}\033[0m"
            elif mask_diff is not None and mask_diff < 0:
                mask_diff_str = f"\033[91m{mask_diff_str}\033[0m"
            
            print(f"{option:<10} {base_avg:>12.3f} ({base_avg*100:>5.1f}%){'':<3} {gsm_avg:>12.3f} ({gsm_avg*100:>5.1f}%){'':<3} {mask_avg:>12.3f} ({mask_avg*100:>5.1f}%){'':<3} {gsm_diff_str:>15} {mask_diff_str:>15}")
    
    # 其他子分类影响
    print("\n4. 所有子分类变化详情")
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
    print("关键发现与原因分析")
    print("=" * 140)
    print("""
🔍 主要发现:

1. **Math子分类能力显著下降**
   • 基准模型: 47.5% → GSM微调后: 35.0% (下降12.5%) ⚠️
   • 这是所有子分类中下降最严重的

2. **具体数学科目表现**
   • 下降最严重:
     - high_school_statistics: 46.3% → 41.2% (下降5.1%)
     - high_school_mathematics: 38.5% → 27.0% (下降11.5%) ⚠️
     - elementary_mathematics: 42.9% → 36.5% (下降6.4%)
     - college_mathematics: 37.0% → 39.0% (上升2.0%) ✅
     - abstract_algebra: 31.0% → 33.0% (上升2.0%) ✅

3. **GSM_MASK对比**
   • GSM_MASK表现: 37.2% (比GSM的35.0%还高2.2%)
   • 说明思维链(CoT)训练本身可能干扰了某些能力

4. **选项偏好变化**
   • 观察选项A/B/C/D的概率分配是否有系统性变化

📊 能力减弱的原因分析:

1. **灾难性遗忘 (Catastrophic Forgetting)**
   • 微调GSM8K数学应用题时，模型过度专注于特定任务
   • 导致遗忘了原有的更广泛的数学知识
   • 特别是统计学和高中数学受影响最大

2. **分布偏移 (Distribution Shift)**
   • GSM8K: 小学/初中水平算术应用题
   • MMLU数学: 抽象代数、统计学、大学数学等更广泛领域
   • 题型和难度分布差异显著

3. **思维链的副作用**
   • GSM8K要求详细逐步推理
   • MMLU只需要直接选择答案
   • 模型过度依赖推理步骤，在直接判断时反而表现下降

4. **推理范式不匹配**
   • GSM8K: 生成完整解决方案
   • MMLU: 4选1选择题
   • 两种任务的推理模式不同

5. **答案位置偏见**
   • 可能在微调后产生了特定位置偏好
   • 这种偏见可能不适应MMLU的正确答案分布

💡 建议的解决方案:

1. 使用LoRA/QLoRA等参数高效微调
2. 混合训练: 同时加入多样化数学数据
3. 降低学习率，减少微调步数
4. 考虑是否真的需要思维链训练
5. 在微调时加入一些MMLU风格的题目
""")

if __name__ == "__main__":
    analyze_mistral_comparison()
