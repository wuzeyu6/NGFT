import json
import os

def modify_json_instruction(file_path):
    """
    读取JSON文件，将每个元素的instruction字段改为"hello"，并保存回原文件
    
    参数:
        file_path (str): JSON文件的路径
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return
    
    try:
        # 1. 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            # 加载JSON数据（支持列表/字典两种常见根结构）
            data = json.load(f)
        
        # 2. 遍历并修改instruction字段
        if isinstance(data, list):
            # 情况1：JSON根是列表（最常见，多个元素）
            for idx, item in enumerate(data):
                if isinstance(item, dict) and 'instruction' in item:
                    prompt = """
                    You are an expert math assistant. Your role is to provide step-by-step calculations for each problem and deliver the correct final answer. Each solution should be logically structured, with no extra commentary or deviation from the required steps. \
Your responses must be concise, accurate, and in the exact format specified below. Your sole focus should be on solving the problem as efficiently as possible. Do not include any extraneous information.\n\
### Guidelines for your response:\n\
1. Your response must contain only step-by-step calculations and the final answer.\n\
2. The final output **must** be formatted as: \\boxed{<number>}.\n\nReplace `<number>` with the correct final result (either an integer or a floating-point number). 
3. Do not add any commentary, questions, greetings, or extra remarks.\n\
4. Ensure your calculations are clear, concise, and correct, but only include the steps required to arrive at the final answer.\n\
Please answer each question step by step and provide the final answer following the instructions below.\n\
                    """
                    data[idx]['instruction'] = prompt
                    print(f"已修改第 {idx+1} 个元素的instruction字段")
                elif isinstance(item, dict):
                    print(f"第 {idx+1} 个元素无instruction字段，跳过")
                else:
                    print(f"第 {idx+1} 个元素不是字典类型，跳过")
        
        elif isinstance(data, dict):
            # 情况2：JSON根是字典（单个元素）
            if 'instruction' in data:
                prompt = """You are an expert math assistant. Your role is to provide step-by-step calculations for each problem and deliver the correct final answer. Each solution should be logically structured, with no extra commentary or deviation from the required steps. \
Your responses must be concise, accurate, and in the exact format specified below. Your sole focus should be on solving the problem as efficiently as possible. Do not include any extraneous information.\n\
### Guidelines for your response:\n\
1. Your response must contain only step-by-step calculations and the final answer.\n\
2. The final output **must** be formatted as: #### <number>.\n\
Replace `<number>` with the correct final result (either an integer or a floating-point number). No deviations or alternative formats are allowed.\n\
3. Do not add any commentary, questions, greetings, or extra remarks.\n\
4. Ensure your calculations are clear, concise, and correct, but only include the steps required to arrive at the final answer.\n\
Please answer each question step by step and provide the final answer following the instructions below.\n\
                    """
                data[idx]['instruction'] = prompt
                print("已修改根字典的instruction字段")
            else:
                print("根字典无instruction字段，跳过")
        
        else:
            print("错误：JSON根结构既不是列表也不是字典，无法处理")
            return
        
        # 3. 保存修改后的数据回原文件（覆盖写入）
        with open(file_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False：保留中文等非ASCII字符
            # indent=4：格式化输出，便于阅读
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"成功！已修改 {file_path} 中的instruction字段为'hello'")
    
    except json.JSONDecodeError:
        print(f"错误：{file_path} 不是有效的JSON文件")
    except PermissionError:
        print(f"错误：没有权限读取/写入 {file_path}")
    except Exception as e:
        print(f"意外错误：{str(e)}")

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 请替换为你的JSON文件路径（绝对路径/相对路径均可）
    json_file_path = "/mnt/shared-storage-user/liushudong/wuzeyu/Neuron/llama_factory/eval_data/math_test.json"  # 例如："./data/test.json" 或 "C:/data/test.json"
    
    # 执行修改操作
    modify_json_instruction(json_file_path)