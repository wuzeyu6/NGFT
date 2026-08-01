
import pandas as pd
import json

def main():
    # 加载训练集（两个parquet文件）
    train_files = [
        "/Users/wu/Desktop/llama_factory/tests/train-00000-of-00002.parquet",
        "/Users/wu/Desktop/llama_factory/tests/train-00001-of-00002.parquet"
    ]
    
    # 读取并合并训练集
    train_dfs = [pd.read_parquet(f) for f in train_files]
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    # 加载测试集
    test_file = "/Users/wu/Desktop/llama_factory/tests/test-00000-of-00001.parquet"
    test_df = pd.read_parquet(test_file)
    
    # 转换为列表
    train_list = train_df.to_dict(orient="records")
    test_list = test_df.to_dict(orient="records")
    
    # 保存为JSON
    with open("/Users/wu/Desktop/llama_factory/DISC-Law-SFT-Alpaca-Train.json", "w", encoding="utf-8") as f:
        json.dump(train_list, f, ensure_ascii=False, indent=2)
    
    with open("/Users/wu/Desktop/llama_factory/DISC-Law-SFT-Alpaca-Test.json", "w", encoding="utf-8") as f:
        json.dump(test_list, f, ensure_ascii=False, indent=2)
    
    print(f"训练集已保存，共 {len(train_list)} 条数据")
    print(f"测试集已保存，共 {len(test_list)} 条数据")

if __name__ == "__main__":
    main()

