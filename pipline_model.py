import os
import logging

# 设置日志配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("error_llama3_Bio.log"),
                        logging.StreamHandler()
                    ])

model_name = 'Mistral-7B-Instruct'
model_path = '/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a'

# 定义3组命令（训练→评估→删除），每组包含3条命令
commands_group = [
    [
        ['python', "eval/mmlu_eval.py", "--save_dir",
         "eval_result/mmlu/base_model/{}".format(model_name), '--model_name_or_path',
         model_path],
        ['python', "eval/tydiqa_eval.py", "--save_dir",
         "eval_result/tydiqa/base_model/{}".format(model_name), '--model_name_or_path',
         model_path],
        ['python', "eval/bbh_eval.py", "--save_dir",
         "eval_result/bbh/base_model/{}".format(model_name), '--model_name_or_path',
         model_path]
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