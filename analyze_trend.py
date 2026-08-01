#!/usr/bin/env python3
import csv
from collections import defaultdict

def analyze_trend(file_path):
    data = []
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            a_prob = float(row[7])
            b_prob = float(row[8])
            c_prob = float(row[9])
            d_prob = float(row[10])
            data.append({
                'idx': i,
                'A': a_prob,
                'B': b_prob,
                'C': c_prob,
                'D': d_prob
            })
    
    print("=== 趋势分析 ===\n")
    
    # 分成前半和后半
    mid = len(data) // 2
    first_half = data[:mid]
    second_half = data[mid:]
    
    print(f"总题数: {len(data)}")
    print(f"前半部分 ({mid} 题):")
    for choice in ['A', 'B', 'C', 'D']:
        avg = sum(x[choice] for x in first_half) / mid
        print(f"  选项 {choice}: {avg:.4f} ({avg*100:.1f}%)")
    
    print(f"\n后半部分 ({len(data)-mid} 题):")
    for choice in ['A', 'B', 'C', 'D']:
        avg = sum(x[choice] for x in second_half) / (len(data)-mid)
        print(f"  选项 {choice}: {avg:.4f} ({avg*100:.1f}%)")
    
    print(f"\n变化（后半 - 前半）:")
    for choice in ['A', 'B', 'C', 'D']:
        first_avg = sum(x[choice] for x in first_half) / mid
        second_avg = sum(x[choice] for x in second_half) / (len(data)-mid)
        diff = second_avg - first_avg
        sign = "+" if diff > 0 else ""
        print(f"  选项 {choice}: {sign}{diff:.4f} ({sign}{diff*100:.1f}%)")
    
    # 看每20题的移动平均
    print(f"\n=== 每20题的平均概率 ===")
    window = 20
    for i in range(0, len(data), window):
        end = min(i + window, len(data))
        chunk = data[i:end]
        print(f"\n题 {i+1}-{end}:")
        for choice in ['A', 'B', 'C', 'D']:
            avg = sum(x[choice] for x in chunk) / len(chunk)
            print(f"  {choice}: {avg:.3f}")

if __name__ == "__main__":
    file_path = "/Users/wu/Desktop/llama_factory/eval_result/mmlu/GSM/GSM_Mistral-7B-Instruct_full/high_school_statistics.csv"
    analyze_trend(file_path)
