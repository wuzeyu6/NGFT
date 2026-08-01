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

def get_metrics(metrics_path):
    """读取metrics.json"""
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def analyze_all_models():
    base_path = "/Users/wu/Desktop/llama_factory/eval_result/mmlu"
    
    # 先收集所有GSM和GSM_MASK的模型
    gsm_configs = set()
    gsm_mask_configs = set()
    
    for finetune_type in ['GSM', 'GSM_MASK']:
        ft_path = os.path.join(base_path, finetune_type)
        if not os.path.isdir(ft_path):
            continue
        
        for config in os.listdir(ft_path):
            if finetune_type == 'GSM':
                gsm_configs.add(config)
            else:
                gsm_mask_configs.add(config)
    
    # 找到配对的模型（配置名在两个文件夹都存在）
    paired_models = []
    for config in gsm_configs:
        if config in gsm_mask_configs:
            paired_models.append(config)
    
    print("=" * 120)
    print("GSM8K 微调影响分析 - GSM vs GSM_MASK 对比")
    print("=" * 120)
    print(f"\n找到 {len(paired_models)} 个配对模型进行对比\n")
    
    # 分析每个配对
    results_by_model = defaultdict(list)
    
    for gsm_model, mask_model in paired_models:
        # 提取信息
        parts = gsm_model.split('_')
        model_name = '_'.join(parts[1:-1])
        finetune_rate = parts[-1]
        
        # 获取GSM结果
        gsm_path = os.path.join(base_path, 'GSM', gsm_model)
        gsm_metrics = get_metrics(os.path.join(gsm_path, 'metrics.json'))
        gsm_accs = {}
        for subj in MATH_SUBJECTS:
            gsm_accs[subj] = calculate_accuracy(os.path.join(gsm_path, f"{subj}.csv"))
        
        # 获取GSM_MASK结果
        mask_path = os.path.join(base_path, 'GSM_MASK', mask_model)
        mask_metrics = get_metrics(os.path.join(mask_path, 'metrics.json'))
        mask_accs = {}
        for subj in MATH_SUBJECTS:
            mask_accs[subj] = calculate_accuracy(os.path.join(mask_path, f"{subj}.csv"))
        
        # 存储结果
        key = f"{model_name}_{finetune_rate}"
        results_by_model[model_name].append({
            'rate': finetune_rate,
            'gsm': {
                'metrics': gsm_metrics,
                'accs': gsm_accs
            },
            'mask': {
                'metrics': mask_metrics,
                'accs': mask_accs
            }
        })
    
    # 按模型显示对比
    for model_name in sorted(results_by_model.keys()):
        print(f"\n\n{'=' * 120}")
        print(f"模型: {model_name}")
        print(f"{'=' * 120}")
        
        # 按微调率排序
        model_results = sorted(results_by_model[model_name], 
                               key=lambda x: (x['rate'] == 'full', x['rate']))
        
        for res in model_results:
            rate = res['rate']
            gsm = res['gsm']
            mask = res['mask']
            
            print(f"\n--- 微调率: {rate} ---")
            
            # 总体Math对比
            gsm_math = gsm['metrics'].get('subcat_acc', {}).get('math') if gsm['metrics'] else None
            mask_math = mask['metrics'].get('subcat_acc', {}).get('math') if mask['metrics'] else None
            
            if gsm_math is not None and mask_math is not None:
                diff = gsm_math - mask_math
                sign = '+' if diff > 0 else ''
                print(f"  总体Math:    GSM={gsm_math:>5.1%}  GSM_MASK={mask_math:>5.1%}  差异={sign}{diff:>+.1%}")
            
            # 各科目对比
            print(f"\n  各数学科目:")
            print(f"  {'科目':<25} {'GSM':<10} {'GSM_MASK':<10} {'差异':<10}")
            print(f"  {''*25} {''*10} {''*10} {''*10}")
            
            for subj in MATH_SUBJECTS:
                gsm_acc = gsm['accs'].get(subj)
                mask_acc = mask['accs'].get(subj)
                
                if gsm_acc is not None and mask_acc is not None:
                    diff = gsm_acc - mask_acc
                    sign = '+' if diff > 0 else ''
                    diff_str = f"{sign}{diff:>+.1%}"
                    if diff > 0:
                        diff_str = f"\033[92m{diff_str}\033[0m"  # 绿色
                    elif diff < 0:
                        diff_str = f"\033[91m{diff_str}\033[0m"  # 红色
                    print(f"  {subj:<25} {gsm_acc:>5.1%}{'':<5} {mask_acc:>5.1%}{'':<5} {diff_str}")
    
    print("\n\n" + "=" * 120)
    print("总结与分析")
    print("=" * 120)
    print("\n关键观察:")
    print("1. GSM vs GSM_MASK: GSM_MASK代表相同数据但思考过程被遮蔽")
    print("2. 如果GSM表现 < GSM_MASK，说明微调GSM8K可能导致了能力退化")
    print("3. 红色数字 = GSM表现不如GSM_MASK（能力退化）")
    print("4. 绿色数字 = GSM表现优于GSM_MASK（能力提升）")

if __name__ == "__main__":
    analyze_all_models()
