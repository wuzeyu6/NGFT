#!/usr/bin/env python3
import csv
from collections import defaultdict

def analyze_csv(file_path):
    choice_probs = defaultdict(list)
    correct_counts = defaultdict(int)
    wrong_counts = defaultdict(int)
    total = 0
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            total += 1
            correct = row[6]
            a_prob = float(row[7])
            b_prob = float(row[8])
            c_prob = float(row[9])
            d_prob = float(row[10])
            
            choice_probs['A'].append(a_prob)
            choice_probs['B'].append(b_prob)
            choice_probs['C'].append(c_prob)
            choice_probs['D'].append(d_prob)
            
            if correct == 'True':
                correct_counts['A'] += 1 if a_prob > b_prob and a_prob > c_prob and a_prob > d_prob else 0
                correct_counts['B'] += 1 if b_prob > a_prob and b_prob > c_prob and b_prob > d_prob else 0
                correct_counts['C'] += 1 if c_prob > a_prob and c_prob > b_prob and c_prob > d_prob else 0
                correct_counts['D'] += 1 if d_prob > a_prob and d_prob > b_prob and d_prob > c_prob else 0
            else:
                wrong_counts['A'] += 1 if a_prob > b_prob and a_prob > c_prob and a_prob > d_prob else 0
                wrong_counts['B'] += 1 if b_prob > a_prob and b_prob > c_prob and b_prob > d_prob else 0
                wrong_counts['C'] += 1 if c_prob > a_prob and c_prob > b_prob and c_prob > d_prob else 0
                wrong_counts['D'] += 1 if d_prob > a_prob and d_prob > b_prob and d_prob > c_prob else 0
    
    print(f"=== 统计结果 ({total} 道题) ===\n")
    
    print("1. 各选项平均概率:")
    for choice in ['A', 'B', 'C', 'D']:
        avg = sum(choice_probs[choice]) / len(choice_probs[choice])
        print(f"   选项 {choice}: {avg:.4f} ({avg*100:.1f}%)")
    
    print("\n2. 各选项被选为最高概率的次数:")
    for choice in ['A', 'B', 'C', 'D']:
        total_chosen = correct_counts[choice] + wrong_counts[choice]
        print(f"   选项 {choice}: {total_chosen} 次 ({total_chosen/total*100:.1f}%)")
    
    print("\n3. 正确/错误分布:")
    for choice in ['A', 'B', 'C', 'D']:
        print(f"   选项 {choice}: 正确 {correct_counts[choice]} 次, 错误 {wrong_counts[choice]} 次")

if __name__ == "__main__":
    file_path = "/Users/wu/Desktop/llama_factory/eval_result/mmlu/GSM/GSM_Mistral-7B-Instruct_full/high_school_statistics.csv"
    analyze_csv(file_path)
