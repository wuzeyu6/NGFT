
from transformers import AutoTokenizer

def analyze_token_encoding():
    # 使用Mistral-7B-Instruct-v0.3的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a")
    
    options = ["A", "B", "C", "D"]
    
    print("选项Token编码分析:")
    print("="*80)
    
    for option in options:
        # 编码选项
        tokens = tokenizer.encode(option, add_special_tokens=False)
        token_str = tokenizer.decode(tokens)
        
        print(f"\n选项 '{option}':")
        print(f"  Token IDs: {tokens}")
        print(f"  Token string: '{token_str}'")
        print(f"  Token数量: {len(tokens)}")
        
        # 查看token的详细信息
        for token_id in tokens:
            token_text = tokenizer.convert_ids_to_tokens(token_id)
            print(f"    - Token ID {token_id}: '{token_text}'")
    
    print("\n" + "="*80)
    print("\n分析带空格和不带空格的情况:")
    
    for option in options:
        # 测试 " A", " B", " C", " D"（带前置空格）
        tokens_with_space = tokenizer.encode(" " + option, add_special_tokens=False)
        print(f"\n' {option}':")
        print(f"  Token IDs: {tokens_with_space}")
        print(f"  Decoded: '{tokenizer.decode(tokens_with_space)}'")
        
        for token_id in tokens_with_space:
            token_text = tokenizer.convert_ids_to_tokens(token_id)
            print(f"    - Token ID {token_id}: '{token_text}'")

if __name__ == "__main__":
    analyze_token_encoding()
