#!/usr/bin/env python3
import json
from collections import Counter

with open("/Users/wu/Desktop/llama_factory/data/BioInstruct_train.json") as f:
    data = json.load(f)

first_chars = Counter()
first_words = Counter()

for item in data[:1000]:
    output = item["output"].strip()
    if output:
        first_char = output[0].upper()
        first_chars[first_char] += 1
        
        first_word = output.split()[0] if len(output.split()) > 0 else ""
        first_words[first_word] += 1

print("Top 20 首字母:")
for char, count in first_chars.most_common(20):
    print(f"{char}: {count} ({count/len(data[:1000])*100:.1f}%)")

print("\nTop 20 首词:")
for word, count in first_words.most_common(20):
    print(f"{word}: {count} ({count/len(data[:1000])*100:.1f}%)")

print("\n示例 output (前50个):")
for item in data[:50]:
    output = item["output"].strip()
    if output:
        first_char = output[0].upper()
        print(f"[{first_char}] {output[:50]}...")
