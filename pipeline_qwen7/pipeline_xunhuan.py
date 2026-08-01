import os
import logging
import subprocess

# 设置日志配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("error_qwen7_Bio.log"),
                        logging.StreamHandler()
                    ])

# 定义3组命令（训练→评估→删除），每组包含3条命令
# commands_group = [
#     # [
#     #     ['llamafactory-cli', 'train', 'examples/train_full_qwen/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.01.yaml'],
#     #     ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.01'],
#     #     # ['python', "eval/mmlu_eval.py", "--save_dir",
#     #     #  "eval_result/mmlu/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.01", '--model_name_or_path',
#     #     #  'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.01'],
#     #     # ['python', "eval/tydiqa_eval.py", "--save_dir",
#     #     #  "eval_result/tydiqa/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.01", '--model_name_or_path',
#     #     #  'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.01'],
#     #     # ['python', "eval/bbh_eval.py", "--save_dir",
#     #     #  "eval_result/bbh/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.01", '--model_name_or_path',
#     #     #  'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.01'],
#     #     ['rm', '-rf', 'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.01']
#     # ],
#     # [
#     #     ['llamafactory-cli', 'train', 'examples/train_full_qwen/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.05.yaml'],
#     #     ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.05'],
#     #     # ['python', "eval/mmlu_eval.py", "--save_dir",
#     #     #  "eval_result/mmlu/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.05", '--model_name_or_path',
#     #     #  'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.05'],
#     #     # ['python', "eval/tydiqa_eval.py", "--save_dir",
#     #     #  "eval_result/tydiqa/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.05", '--model_name_or_path',
#     #     #  'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.05'],
#     #     # ['python', "eval/bbh_eval.py", "--save_dir",
#     #     #  "eval_result/bbh/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.05", '--model_name_or_path',
#     #     #  'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.05'],
#     #     ['rm', '-rf', 'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.05']
#     # ],
#     # [
#     #     ['llamafactory-cli', 'train', 'examples/train_full_qwen/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.10.yaml'],
#     #     ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.10'],
#     #     # ['python', "eval/mmlu_eval.py", "--save_dir",
#     #     #  "eval_result/mmlu/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.10", '--model_name_or_path',
#     #     #  'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.10'],
#     #     # ['python', "eval/tydiqa_eval.py", "--save_dir",
#     #     #  "eval_result/tydiqa/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.10", '--model_name_or_path',
#     #     #  'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.10'],
#     #     # ['python', "eval/bbh_eval.py", "--save_dir",
#     #     #  "eval_result/bbh/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.10", '--model_name_or_path',
#     #     #  'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.10'],
#     #     ['rm', '-rf', 'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_0.10']
#     # ],
#     # [
#     #     ['llamafactory-cli', 'train', 'examples/train_full_qwen/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_full.yaml'],
#     #     ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_full'],
#     #     # ['python', "eval/mmlu_eval.py", "--save_dir",
#     #     #  "eval_result/mmlu/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_full", '--model_name_or_path',
#     #     #  'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_full'],
#     #     # ['python', "eval/tydiqa_eval.py", "--save_dir",
#     #     #  "eval_result/tydiqa/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_full", '--model_name_or_path',
#     #     #  'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_full'],
#     #     # ['python', "eval/bbh_eval.py", "--save_dir",
#     #     #  "eval_result/bbh/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_full", '--model_name_or_path',
#     #     #  'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_full'],
#     #     ['rm', '-rf', 'model_result/BioInstruct/BioInstruct_Qwen2.5-7B-Instruct_full']
#     # ],
#     # [
#     #     ['llamafactory-cli', 'train', 'examples/train_full_qwen/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.01.yaml'],
#     #     ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.01'],
#     #     # ['python', "eval/mmlu_eval.py", "--save_dir",
#     #     #  "eval_result/mmlu/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.01", '--model_name_or_path',
#     #     #  'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.01'],
#     #     # ['python', "eval/tydiqa_eval.py", "--save_dir",
#     #     #  "eval_result/tydiqa/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.01", '--model_name_or_path',
#     #     #  'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.01'],
#     #     # ['python', "eval/bbh_eval.py", "--save_dir",
#     #     #  "eval_result/bbh/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.01", '--model_name_or_path',
#     #     #  'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.01'],
#     #     ['rm', '-rf', 'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.01']
#     # ],
#     # [
#     #     ['llamafactory-cli', 'train', 'examples/train_full_qwen/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.05.yaml'],
#     #     ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.05'],
#     #     # ['python', "eval/mmlu_eval.py", "--save_dir",
#     #     #  "eval_result/mmlu/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.05", '--model_name_or_path',
#     #     #  'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.05'],
#     #     # ['python', "eval/tydiqa_eval.py", "--save_dir",
#     #     #  "eval_result/tydiqa/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.05", '--model_name_or_path',
#     #     #  'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.05'],
#     #     # ['python', "eval/bbh_eval.py", "--save_dir",
#     #     #  "eval_result/bbh/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.05", '--model_name_or_path',
#     #     #  'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.05'],
#     #     ['rm', '-rf', 'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.05']
#     # ],
#     # [
#     #     ['llamafactory-cli', 'train', 'examples/train_full_qwen/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.10.yaml'],
#     #     ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.10'],
#     #     # ['python', "eval/mmlu_eval.py", "--save_dir",
#     #     #  "eval_result/mmlu/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.10", '--model_name_or_path',
#     #     #  'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.10'],
#     #     # ['python', "eval/tydiqa_eval.py", "--save_dir",
#     #     #  "eval_result/tydiqa/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.10", '--model_name_or_path',
#     #     #  'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.10'],
#     #     # ['python', "eval/bbh_eval.py", "--save_dir",
#     #     #  "eval_result/bbh/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.10", '--model_name_or_path',
#     #     #  'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.10'],
#     #     ['rm', '-rf', 'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-7B-Instruct_0.10']
#     # ],
#     [
#         ['CUDA_VISIBLE_DEVICES=4,5,6', 'python', "eval/mmlu_eval.py", "--save_dir",
#          "eval_result/mmlu/Qwen2.5-7B-Instruct_full", '--model_name_or_path',
#          '/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'],
#     ],
# ]
commands_group = []

for j in range(300):
    commands_group.append([
        ['python', "all_eval_1.py", '--model_path',
         '/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a'],
    ])
# 循环执行3组命令（每组内部依次执行训练→评估→删除）
for group_idx, command_group in enumerate(commands_group, start=1):
    print(f"=== 开始执行第 {group_idx} 组命令 ===")
    for cmd in command_group:
        try:
            # 执行单条命令，check=True 会在命令返回非0退出码时抛出异常
            subprocess.run(cmd, check=True,stderr=subprocess.PIPE,
            text=True)
            print(f"成功执行命令: {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败: {' '.join(cmd)}，错误: {e}")
            print(f"错误信息: {e.stderr}")  # 这里会显示具体的错误原因
            with open("error.log", "a", encoding="utf-8") as f:
                f.write(f"命令执行失败: {' '.join(cmd)}，错误: {e.stderr}")
            # 可根据需求决定是否终止后续命令，若需终止直接 break 或 return
    print(f"=== 第 {group_idx} 组命令执行完毕 ===\n")