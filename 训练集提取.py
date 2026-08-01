import json
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def main():
    # 设置随机种子
    random.seed(42)
    
    # 加载数据
    print("正在加载数据...")
    with open('/Users/wu/Desktop/llama_factory/DISC-Law-SFT-Alpaca-Test_2000_filtered.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    with open('/Users/wu/Desktop/llama_factory/DISC-Law-SFT-Alpaca-Train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 过滤只保留 legal_question_answering 类型
    def filter_legal_qa(data):
        import re
        filtered = []
        for item in data:
            match = re.match(r'^legal_question_answering', item['id'])
            if match:
                filtered.append(item)
        return filtered
    
    test_data = filter_legal_qa(test_data)
    train_data = filter_legal_qa(train_data)
    
    print(f"Test数据: {len(test_data)} 条 (仅 legal_question_answering)")
    print(f"Train数据: {len(train_data)} 条 (仅 legal_question_answering)")
    
    # 加载模型
    print("\n正在加载模型...")
    model = SentenceTransformer('shibing624/text2vec-base-chinese')
    
    # 预处理train数据：分离有input和无input的
    print("\n正在预处理train数据...")
    train_with_input = []
    train_without_input = []
    
    for item in train_data:
        if item.get('input', '').strip():
            train_with_input.append(item)
        else:
            train_without_input.append(item)
    
    print(f"Train有input: {len(train_with_input)} 条")
    print(f"Train无input: {len(train_without_input)} 条")
    
    # 预计算所有train数据的embeddings
    print("\n正在计算train数据的embeddings...")
    
    def compute_embeddings(items, model):
        instructions = [item['instruction'] for item in items]
        outputs = [item['output'] for item in items]
        
        instruction_embeddings = model.encode(instructions, batch_size=64, show_progress_bar=True)
        output_embeddings = model.encode(outputs, batch_size=64, show_progress_bar=True)
        
        return instruction_embeddings, output_embeddings
    
    print("\n处理有input的train数据...")
    train_instr_emb_with_input, train_out_emb_with_input = compute_embeddings(train_with_input, model)
    
    print("\n处理无input的train数据...")
    train_instr_emb_without_input, train_out_emb_without_input = compute_embeddings(train_without_input, model)
    
    # 准备数据引用
    train_data_refs = {
        'with_input': {
            'items': train_with_input,
            'instr_emb': train_instr_emb_with_input,
            'out_emb': train_out_emb_with_input
        },
        'without_input': {
            'items': train_without_input,
            'instr_emb': train_instr_emb_without_input,
            'out_emb': train_out_emb_without_input
        }
    }
    
    # 多线程处理test数据
    print("\n正在多线程处理test数据...")
    results = []
    lock = threading.Lock()
    
    def process_single_test(test_item):
        """处理单条test数据"""
        # 判断test数据是否有input
        has_input = bool(test_item.get('input', '').strip())
        
        # 选择对应的train数据
        if has_input:
            ref = train_data_refs['with_input']
        else:
            ref = train_data_refs['without_input']
        
        candidate_train = ref['items']
        train_instr_emb = ref['instr_emb']
        train_out_emb = ref['out_emb']
        
        # 计算test数据的embeddings
        test_instr_emb = model.encode([test_item['instruction']])
        test_out_emb = model.encode([test_item['output']])
        
        # 计算相似度
        instr_similarities = cosine_similarity(test_instr_emb, train_instr_emb)[0]
        out_similarities = cosine_similarity(test_out_emb, train_out_emb)[0]
        
        # 总相似度 = instruction相似度 + output相似度
        total_similarities = instr_similarities + out_similarities
        
        # 找出top 5最相似的
        top_indices = np.argsort(total_similarities)[::-1][:5]
        
        # 生成结果
        item_results = []
        for rank, idx in enumerate(top_indices):
            result_item = {
                'test_id': test_item['id'],
                'test_instruction': test_item['instruction'],
                'test_input': test_item.get('input', ''),
                'test_output': test_item['output'],
                'train_id': candidate_train[idx]['id'],
                'train_instruction': candidate_train[idx]['instruction'],
                'train_input': candidate_train[idx].get('input', ''),
                'train_output': candidate_train[idx]['output'],
                'instruction_similarity': float(instr_similarities[idx]),
                'output_similarity': float(out_similarities[idx]),
                'total_similarity': float(total_similarities[idx]),
                'rank': rank + 1
            }
            item_results.append(result_item)
        
        return item_results
    
    # 使用线程池处理
    max_workers = 4  # 可以根据CPU核心数调整
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {executor.submit(process_single_test, item): i for i, item in enumerate(test_data)}
        
        # 收集结果
        with tqdm(total=len(test_data), desc="处理进度") as pbar:
            for future in as_completed(future_to_idx):
                item_results = future.result()
                with lock:
                    results.extend(item_results)
                pbar.update(1)
    
    # 保存结果
    print(f"\n正在保存结果...共 {len(results)} 条数据")
    output_file = "/Users/wu/Desktop/llama_factory/similarity_results_10000.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"完成！结果已保存至: {output_file}")
    
    # 统计信息
    print("\n统计信息:")
    print(f"总匹配数: {len(results)}")
    print(f"平均instruction相似度: {np.mean([r['instruction_similarity'] for r in results]):.4f}")
    print(f"平均output相似度: {np.mean([r['output_similarity'] for r in results]):.4f}")
    print(f"平均总相似度: {np.mean([r['total_similarity'] for r in results]):.4f}")

if __name__ == "__main__":
    main()