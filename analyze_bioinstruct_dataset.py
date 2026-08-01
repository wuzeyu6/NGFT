
import json
from collections import Counter
import re

def analyze_bioinstruct_dataset():
    with open("/Users/wu/Desktop/llama_factory/data/BioInstruct_train.json") as f:
        data = json.load(f)
    
    print(f"数据集总条数: {len(data)}")
    print("\n" + "="*80)
    
    # 分析output的长度、首词、首字母、首句等
    output_lengths = []
    first_words = Counter()
    first_chars = Counter()
    first_sentences = Counter()
    starts_with_letter = Counter()
    starts_with_article = Counter()  # The, A, An
    
    for item in data:
        output = item["output"].strip()
        if not output:
            continue
            
        # 长度
        output_lengths.append(len(output))
        
        # 首字母（大写）
        first_char = output[0].upper()
        first_chars[first_char] += 1
        
        # 首词
        words = output.split()
        if words:
            first_word = words[0]
            first_words[first_word] += 1
            
            # 检查是否以冠词开头
            if first_word.lower() in ["the", "a", "an"]:
                starts_with_article[first_word.lower()] += 1
                
            # 检查是否以单个字母开头（如 "A.", "B.", "A:", "B:"等）
            if len(first_word) == 1 and first_word.isalpha():
                starts_with_letter[first_word] += 1
            elif len(first_word) >= 2 and first_word[1] in [".", ":"] and first_word[0].isalpha():
                starts_with_letter[first_word[0]] += 1
        
        # 首句
        first_sentence = output.split(".")[0].strip() if "." in output else output
        first_sentences[first_sentence] += 1
    
    # 打印统计结果
    print("\nOutput长度统计:")
    print(f"平均长度: {sum(output_lengths)/len(output_lengths):.1f}")
    print(f"最短长度: {min(output_lengths)}")
    print(f"最长长度: {max(output_lengths)}")
    
    print("\n" + "="*80)
    print("\nTop 20 首词分布:")
    total = sum(first_words.values())
    for word, count in first_words.most_common(20):
        print(f"{word}: {count} ({count/total*100:.1f}%)")
    
    print("\n" + "="*80)
    print("\n首字母分布（Top 10）:")
    total = sum(first_chars.values())
    for char, count in first_chars.most_common(10):
        print(f"{char}: {count} ({count/total*100:.1f}%)")
    
    print("\n" + "="*80)
    print("\n以冠词开头的分布:")
    total = sum(starts_with_article.values())
    for word, count in starts_with_article.most_common():
        print(f"{word}: {count} ({count/total*100:.1f}%)")
    
    print("\n" + "="*80)
    print("\n以单个字母开头的情况:")
    if starts_with_letter:
        total = sum(starts_with_letter.values())
        for char, count in starts_with_letter.most_common():
            print(f"{char}: {count} ({count/total*100:.1f}%)")
    else:
        print("没有发现以单个字母开头的output")
    
    print("\n" + "="*80)
    print("\nTop 10 首句:")
    for sent, count in first_sentences.most_common(10):
        print(f"{sent}: {count}次")
    
    print("\n" + "="*80)
    print("\n前50条数据的output预览:")
    for i, item in enumerate(data[:50]):
        output = item["output"].strip()
        if len(output) > 100:
            print(f"\n{i+1}. {output[:100]}...")
        else:
            print(f"\n{i+1}. {output}")

if __name__ == "__main__":
    analyze_bioinstruct_dataset()
