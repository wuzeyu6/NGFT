import os
import logging

# 设置日志配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("error_qwen7_Bio.log"),
                        logging.StreamHandler()
                    ])

# 定义3组命令（训练→评估→删除），每组包含3条命令
commands_group = [
    [
        ['llamafactory-cli', 'train', 'examples/train_full_qwen_14b/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.01.yaml'],
        ['python', 'delete_checkpoint.py', '--root_dir', "model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.01",'--no-dry-run'],
        ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.01'],
        ['python', "eval/mmlu_eval.py", "--save_dir",
         "eval_result/mmlu/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.01", '--model_name_or_path',
         'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.01'],
        ['python', "eval/tydiqa_eval.py", "--save_dir",
         "eval_result/tydiqa/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.01", '--model_name_or_path',
         'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.01'],
        ['python', "eval/bbh_eval.py", "--save_dir",
         "eval_result/bbh/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.01", '--model_name_or_path',
         'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.01'],
        ['rm', '-rf', 'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.01']
    ],
    [
        ['llamafactory-cli', 'train', 'examples/train_full_qwen_14b/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.05.yaml'],
     ['python', 'delete_checkpoint.py', '--root_dir', "model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.05",'--no-dry-run'],
        ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.05'],
        ['python', "eval/mmlu_eval.py", "--save_dir",
         "eval_result/mmlu/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.05", '--model_name_or_path',
         'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.05'],
        ['python', "eval/tydiqa_eval.py", "--save_dir",
         "eval_result/tydiqa/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.05", '--model_name_or_path',
         'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.05'],
        ['python', "eval/bbh_eval.py", "--save_dir",
         "eval_result/bbh/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.05", '--model_name_or_path',
         'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.05'],
        ['rm', '-rf', 'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.05']
    ],
    [
        ['llamafactory-cli', 'train', 'examples/train_full_qwen_14b/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.10.yaml'],
        ['python', 'delete_checkpoint.py', '--root_dir', "model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.10",'--no-dry-run'],
        ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.10'],
        ['python', "eval/mmlu_eval.py", "--save_dir",
         "eval_result/mmlu/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.10", '--model_name_or_path',
         'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.10'],
        ['python', "eval/tydiqa_eval.py", "--save_dir",
         "eval_result/tydiqa/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.10", '--model_name_or_path',
         'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.10'],
        ['python', "eval/bbh_eval.py", "--save_dir",
         "eval_result/bbh/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.10", '--model_name_or_path',
         'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.10'],
        ['rm', '-rf', 'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_0.10']
    ],
    [
        ['llamafactory-cli', 'train', 'examples/train_full_qwen_14b/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_full.yaml'],
        ['python', 'delete_checkpoint.py', '--root_dir', "model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_full",'--no-dry-run'],
        ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_full'],
        ['python', "eval/mmlu_eval.py", "--save_dir",
         "eval_result/mmlu/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_full", '--model_name_or_path',
         'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_full'],
        ['python', "eval/tydiqa_eval.py", "--save_dir",
         "eval_result/tydiqa/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_full", '--model_name_or_path',
         'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_full'],
        ['python', "eval/bbh_eval.py", "--save_dir",
         "eval_result/bbh/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_full", '--model_name_or_path',
         'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_full'],
        ['rm', '-rf', 'model_result/BioInstruct/BioInstruct_Qwen2.5-14B-Instruct_full']
    ],
    [
        ['llamafactory-cli', 'train', 'examples/train_full_qwen_14b/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.01.yaml'],
        ['python', 'delete_checkpoint.py', '--root_dir', "model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.01",'--no-dry-run'],
        ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.01'],
        ['python', "eval/mmlu_eval.py", "--save_dir",
         "eval_result/mmlu/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.01", '--model_name_or_path',
         'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.01'],
        ['python', "eval/tydiqa_eval.py", "--save_dir",
         "eval_result/tydiqa/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.01", '--model_name_or_path',
         'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.01'],
        ['python', "eval/bbh_eval.py", "--save_dir",
         "eval_result/bbh/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.01", '--model_name_or_path',
         'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.01'],
        ['rm', '-rf', 'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.01']
    ],
    [
        ['llamafactory-cli', 'train', 'examples/train_full_qwen_14b/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.05.yaml'],
        ['python', 'delete_checkpoint.py', '--root_dir', "model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.05",'--no-dry-run'],
        ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.05'],
        ['python', "eval/mmlu_eval.py", "--save_dir",
         "eval_result/mmlu/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.05", '--model_name_or_path',
         'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.05'],
        ['python', "eval/tydiqa_eval.py", "--save_dir",
         "eval_result/tydiqa/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.05", '--model_name_or_path',
         'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.05'],
        ['python', "eval/bbh_eval.py", "--save_dir",
         "eval_result/bbh/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.05", '--model_name_or_path',
         'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.05'],
        ['rm', '-rf', 'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.05']
    ],
    [
        ['llamafactory-cli', 'train', 'examples/train_full_qwen_14b/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.10.yaml'],
        ['python', 'delete_checkpoint.py', '--root_dir', "model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.10",'--no-dry-run'],
        ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.10'],
        ['python', "eval/mmlu_eval.py", "--save_dir",
         "eval_result/mmlu/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.10", '--model_name_or_path',
         'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.10'],
        ['python', "eval/tydiqa_eval.py", "--save_dir",
         "eval_result/tydiqa/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.10", '--model_name_or_path',
         'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.10'],
        ['python', "eval/bbh_eval.py", "--save_dir",
         "eval_result/bbh/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.10", '--model_name_or_path',
         'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.10'],
        ['rm', '-rf', 'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_0.10']
    ],
    [
        ['llamafactory-cli', 'train', 'examples/train_full_qwen_14b/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_full.yaml'],
        ['python', 'delete_checkpoint.py', '--root_dir', "model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_full",'--no-dry-run'],
        ['python', 'all_eval.py', '--model_path', 'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_full'],
        ['python', "eval/mmlu_eval.py", "--save_dir",
         "eval_result/mmlu/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_full", '--model_name_or_path',
         'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_full'],
        ['python', "eval/tydiqa_eval.py", "--save_dir",
         "eval_result/tydiqa/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_full", '--model_name_or_path',
         'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_full'],
        ['python', "eval/bbh_eval.py", "--save_dir",
         "eval_result/bbh/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_full", '--model_name_or_path',
         'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_full'],
        ['rm', '-rf', 'model_result/BioInstruct_MASK/BioInstruct_Qwen2.5-14B-Instruct_full']
    ]
]

# 循环执行3组命令（每组内部依次执行训练→评估→删除）
for group_idx, command_group in enumerate(commands_group, start=1):
    logging.info(f"=== 开始执行第 {group_idx} 组命令 ===")
    for cmd in command_group:
        try:
            logging.info(f"正在执行命令: {' '.join(cmd)}")
            result = os.system(' '.join(cmd))

            if result == 0:
                logging.info(f"成功执行命令: {' '.join(cmd)}")
            else:
                logging.error(f"命令执行失败: {' '.join(cmd)}，返回码: {result}")
                with open("error.log", "a", encoding="utf-8") as f:
                    f.write(f"命令执行失败: {' '.join(cmd)}，返回码: {result}\n")
                # 可根据需求决定是否终止后续命令，若需终止直接 break 或 return
        except Exception as e:
            logging.error(f"命令执行过程中发生异常: {' '.join(cmd)}，异常: {e}")
            with open("error.log", "a", encoding="utf-8") as f:
                f.write(f"命令执行过程中发生异常: {' '.join(cmd)}，异常: {e}\n")

    logging.info(f"=== 第 {group_idx} 组命令执行完毕 ===\n")