#!/usr/bin/env python3
import json
import os
import pandas as pd
from collections import Counter, defaultdict
import numpy as np

print("=== Deep Analysis of BioInstruct Fine-tuning Problem ===\n")

# 1. 加载原始BioInstruct数据集
print("1. Loading original BioInstruct dataset...")
with open("/Users/wu/Desktop/llama_factory/data/BioInstruct_train.json", "r") as f:
    bioinstruct_data = json.load(f)
print(f"   Total examples: {len(bioinstruct_data)}")

# 2. 分析原始数据集output的格式
print("\n2. Analyzing BioInstruct output formats...")
output_lengths = []
first_words = []
first_chars = []
starts_with_number = []
for item in bioinstruct_data[:1000]:
    output = item["output"].strip()
    output_lengths.append(len(output))
    if output:
        words = output.split()
        if words:
            first_word = words[0]
            first_words.append(first_word)
        first_char = output[0].upper()
        first_chars.append(first_char)
        starts_with_number.append(output[0].isdigit())

print(f"   Average output length: {np.mean(output_lengths):.1f}")
print(f"   Top 10 first words: {Counter(first_words).most_common(10)}")
print(f"   Top 10 first chars: {Counter(first_chars).most_common(10)}")
print(f"   Starts with number: {sum(starts_with_number)/len(starts_with_number)*100:.1f}%")

# 3. 加载模型输出结果
print("\n3. Loading MMLU evaluation results...")

base_dir = "/Users/wu/Desktop/llama_factory/eval_result/mmlu"
bioinstruct_result_path = os.path.join(base_dir, "BioInstruct", "BioInstruct_Mistral-7B-Instruct_full")
base_result_path = os.path.join(base_dir, "base_model", "Mistral-7B-Instruct")

# 获取生物相关的科目
bio_subjects = [
    "college_biology", 
    "high_school_biology", 
    "professional_medicine", 
    "medical_genetics", 
    "virology",
    "anatomy",
    "nutrition",
    "clinical_knowledge"
]

# 加载并分析每个科目的结果
print("\n4. Analyzing per-subject results...")
all_bioinstruct_preds = []
all_base_preds = []
all_gts = []

total_bioinstruct_correct = 0
total_bioinstruct_count = 0
total_base_correct = 0
total_base_count = 0

for subject in bio_subjects:
    bioinstruct_file = os.path.join(bioinstruct_result_path, f"{subject}.csv")
    base_file = os.path.join(base_result_path, f"{subject}.csv")
    
    if os.path.exists(bioinstruct_file):
        bioinstruct_df = pd.read_csv(bioinstruct_file)
        base_df = pd.read_csv(base_file)
        
        # 获取预测结果
        choices = ["A", "B", "C", "D"]
        probs_cols = [f"choice{c}_probs" for c in choices]
        
        # 统计BioInstruct模型的预测
        bioinstruct_pred_indices = bioinstruct_df[probs_cols].values.argmax(axis=1)
        bioinstruct_preds = [choices[i] for i in bioinstruct_pred_indices]
        all_bioinstruct_preds.extend(bioinstruct_preds)
        
        bioinstruct_gts = bioinstruct_df.iloc[:, -2].values  # correct列前一列是GT
        all_gts.extend(bioinstruct_gts)
        
        bioinstruct_correct = (bioinstruct_preds == bioinstruct_gts).sum()
        total_bioinstruct_correct += bioinstruct_correct
        total_bioinstruct_count += len(bioinstruct_preds)
        
        # 统计base模型的预测
        base_pred_indices = base_df[probs_cols].values.argmax(axis=1)
        base_preds = [choices[i] for i in base_pred_indices]
        all_base_preds.extend(base_preds)
        
        base_gts = base_df.iloc[:, -2].values
        base_correct = (base_preds == base_gts).sum()
        total_base_correct += base_correct
        total_base_count += len(base_preds)
        
        print(f"\n   {subject}:")
        print(f"     BioInstruct - Acc: {bioinstruct_correct/len(bioinstruct_preds)*100:.1f}%, Preds: {Counter(bioinstruct_preds)}")
        print(f"     Base       - Acc: {base_correct/len(base_preds)*100:.1f}%, Preds: {Counter(base_preds)}")

# 整体统计
print(f"\n5. Overall Results:")
print(f"   BioInstruct - Total: {total_bioinstruct_count}, Correct: {total_bioinstruct_correct}, Acc: {total_bioinstruct_correct/total_bioinstruct_count*100:.1f}%")
print(f"   Base       - Total: {total_base_count}, Correct: {total_base_correct}, Acc: {total_base_correct/total_base_count*100:.1f}%")
print(f"   BioInstruct Preds Distribution: {Counter(all_bioinstruct_preds)}")
print(f"   Base Preds Distribution: {Counter(all_base_preds)}")
print(f"   GT Distribution: {Counter(all_gts)}")

# 6. 详细分析预测B的情况
print(f"\n6. Detailed Analysis of B Predictions:")
b_indices = [i for i, pred in enumerate(all_bioinstruct_preds) if pred == "B"]
print(f"   Total B predictions: {len(b_indices)} ({len(b_indices)/len(all_bioinstruct_preds)*100:.1f}%)")

# 7. 检查MMLU的prompt格式
print(f"\n7. MMLU Prompt Format Analysis:")
print("   Let's look at one example from the MMLU evaluation:")

# 找一个MMLU的测试数据
mmlu_data_dir = "/Users/wu/Desktop/llama_factory/eval/eval/mmlu/data"
dev_file = os.path.join(mmlu_data_dir, "dev", "college_biology_dev.csv")
if os.path.exists(dev_file):
    dev_df = pd.read_csv(dev_file, header=None)
    print(f"\n   College Biology Dev Example (first 3 rows):")
    for i in range(min(3, len(dev_df))):
        print(f"\n   Example {i+1}:")
        print(f"     Question: {dev_df.iloc[i, 0]}")
        for j, c in enumerate(["A", "B", "C", "D"]):
            if j+1 < len(dev_df.columns)-1:
                print(f"     {c}. {dev_df.iloc[i, j+1]}")
        print(f"     Answer: {dev_df.iloc[i, -1]}")

# 8. 检查实际微调时使用的BioInstruct Mistral数据
print(f"\n8. Checking Fine-tuning Data Format:")
bioinstruct_mistral_file = "/Users/wu/Desktop/llama_factory/data/BioInstruct_Mistral-7B-Instruct_0.10.json"
if os.path.exists(bioinstruct_mistral_file):
    with open(bioinstruct_mistral_file, "r") as f:
        bioinstruct_mistral = json.load(f)
    print(f"   BioInstruct Mistral Data - Total examples: {len(bioinstruct_mistral)}")
    print(f"\n   First 3 examples:")
    for i in range(min(3, len(bioinstruct_mistral))):
        item = bioinstruct_mistral[i]
        print(f"\n   Example {i+1}:")
        print(f"     Instruction: {item.get('instruction', '')[:100]}...")
        print(f"     Input: {item.get('input', '')[:100]}...")
        print(f"     Output: {item.get('output', '')[:100]}...")

print("\n=== Analysis Complete ===")
