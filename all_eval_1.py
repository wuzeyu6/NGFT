import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import numpy as np
from eval.eval.MATH.minerva_utils import normalize_final_answer, get_unnormalized_answer, is_equiv


os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=256, help="批量大小")  # 添加批量大小参数
    return parser.parse_args()


from rouge_score import rouge_scorer
import numpy as np
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def _calculate_single_pair(args: Tuple[str, str, bool]) -> Dict[str, float]:
    """单个文本对的ROUGE-L计算（用于并行处理）"""
    candidate, reference, use_stemmer = args
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=use_stemmer)
    scores = scorer.score(reference, candidate)['rougeL']
    return {
        "precision": scores.precision,
        "recall": scores.recall,
        "fmeasure": scores.fmeasure
    }

def calculate_rouge_l_parallel(
        candidates: List[str],
        references: List[str],
        use_stemmer: bool = True,
        verbose: bool = False,
        max_workers: int = None
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    并行计算多个候选文本与参考文本的ROUGE-L分数

    参数:
        candidates: 候选文本列表
        references: 参考文本列表，需与候选文本一一对应
        use_stemmer: 是否使用词干提取器
        verbose: 是否打印详细结果
        max_workers: 最大进程数，默认使用CPU核心数

    返回:
        元组，包含两个元素:
        1. 每个文本对的ROUGE-L分数列表（与输入顺序一致）
        2. 所有分数的汇总统计信息
    """
    # 验证输入长度是否匹配
    if len(candidates) != len(references):
        raise ValueError("候选文本列表与参考文本列表长度必须一致")

    # 准备并行计算的参数
    num_pairs = len(candidates)
    args_list = [(candidates[i], references[i], use_stemmer) for i in range(num_pairs)]

    # 初始化结果列表（保持输入顺序）
    results = [None] * num_pairs

    # 使用多进程并行计算
    max_workers = max_workers or multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务并保留未来对象与索引的映射
        future_to_index = {
            executor.submit(_calculate_single_pair, args): i
            for i, args in enumerate(args_list)
        }

        # 处理完成的任务
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
                if verbose:
                    print(f"已完成 {index + 1}/{num_pairs} 对文本计算")
            except Exception as e:
                print(f"计算第 {index + 1} 对文本时出错: {str(e)}")
                results[index] = None

    # 过滤无效结果
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        raise RuntimeError("所有文本对计算均失败")

    # --- 修改开始 ---
    # 根据您的要求，基于fmeasure去掉两个最低分和两个最高分的样本
    num_to_trim = 8
    if len(valid_results) > num_to_trim:
        # 按fmeasure排序
        sorted_results = sorted(valid_results, key=lambda r: r['fmeasure'])
        # 裁剪样本列表
        trimmed_results = sorted_results[num_to_trim:]
    else:
        # 如果数据点不够多，则不进行裁剪
        trimmed_results = valid_results

    if not trimmed_results:
        raise RuntimeError("裁剪后没有剩余的有效结果")

    # 从裁剪后的样本中提取分数
    precisions = [r['precision'] for r in trimmed_results]
    recalls = [r['recall'] for r in trimmed_results]
    fmeasures = [r['fmeasure'] for r in trimmed_results]

    # 使用裁剪后的数据计算汇总统计
    stats = {
        "avg_precision": np.mean(precisions),
        "avg_recall": np.mean(recalls),
        "avg_fmeasure": np.mean(fmeasures),
        "std_precision": np.std(precisions),
        "std_recall": np.std(recalls),
        "std_fmeasure": np.std(fmeasures),
        "min_precision": np.min(precisions),
        "max_precision": np.max(precisions),
        "min_recall": np.min(recalls),
        "max_recall": np.max(recalls),
        "min_fmeasure": np.min(fmeasures),
        "max_fmeasure": np.max(fmeasures),
        "count": len(valid_results),  # 原始有效样本数
        "trimmed_count": len(trimmed_results), # 裁剪后的样本数
        "failed_count": num_pairs - len(valid_results)
    }
    # --- 修改结束 ---

    return results, stats


def append_to_txt(text, filename, encoding='utf-8', add_newline=True):
    """
    将文本追加到指定的txt文档中（不会覆盖原有内容）

    参数:
        text (str): 要写入的文本内容
        filename (str): 目标txt文件名（包含路径，如无路径则保存在当前目录）
        encoding (str): 文件编码格式，默认utf-8
        add_newline (bool): 是否在每次写入后添加换行符，默认True（使每条内容分行）

    返回:
        bool: 写入成功返回True，失败返回False
    """
    try:
        # 以追加模式打开文件（'a'表示追加，不存在则创建）
        with open(filename, 'a', encoding=encoding) as f:
            # 如果需要换行，在文本后添加换行符
            content = text + '\n' if add_newline else text
            f.write(content)
        return True
    except Exception as e:
        print(f"写入失败：{str(e)}")
        return Falses

def eval_math(model_name_or_path, test_data_path, batch_size):
    example = []
    if torch.cuda.device_count() > 1:
        device_count_ = 4
    else:
        device_count_ = 1
    llm = LLM(model=model_name_or_path, tokenizer=model_name_or_path,
              tokenizer_mode="auto",
              tensor_parallel_size=device_count_)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=1000, temperature=0)

    data = []
    df = json.load(open(test_data_path, "r"))
    for i in range(len(df)):
        for j in range(1):
            data.append(df[i])

    # 修改为批量循环（原单个循环改为批量循环）
    for i in tqdm(range(0, len(data), batch_size)):
        batch_data = data[i:i + batch_size]  # 取当前批次数据
        texts = []
        # 构建批次输入文本
        for item in batch_data:
            message = [
                {"role": "system", "content": item["instruction"]},
                {"role": "user", "content": item["input"]}
            ]
            text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            texts.append(text)

        # 使用vllm进行批量生成
        outputs = llm.generate(texts, sampling_params)

        for idx, output in enumerate(outputs):
            response = output.outputs[0].text
            item = batch_data[idx]  # 当前样本数据
            example.append({
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"],
                "answer": item["answer"],
                "response": response
            })
        print("已经完成了", (i + batch_size)/len(data))

    final_result = []
    for i in range(int(len(example)/1)):
        number = int(1*i)
        answer_cal = []
        for j in range(number,number+1):
            try:

                pattern = r'\\boxed\{(.*?)\}'
                result = re.search(pattern, example[j]["response"])
                answer = result.group(1)
                correct = 1 if is_equiv(answer, example[j]['answer']) else 0
                answer_cal.append(correct)
            except Exception as e:
                print(f"提取不到信息：{e}")
                answer_cal.append(0)

        result_mean = sum(answer_cal)/1
        result_max = max(answer_cal)
        result_min = min(answer_cal)
        final_result.append([result_mean, result_max, result_min])

    mean_mean = np.mean(final_result, axis=0)[0]
    mean_max = np.mean(final_result, axis=0)[1]
    mean_min = np.mean(final_result, axis=0)[2]

    return example, mean_mean, mean_max, mean_min


def eval_gsm(model_name_or_path, test_data_path, batch_size):
    example = []
    if torch.cuda.device_count() > 1:
        device_count_ = 4
    else:
        device_count_ = 1
    llm = LLM(model=model_name_or_path, tokenizer=model_name_or_path,
              tokenizer_mode="auto",
              tensor_parallel_size=device_count_, gpu_memory_utilization=0.7)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=1000, temperature=0)

    data = []
    df = json.load(open(test_data_path, "r"))
    for i in range(len(df)):
        for j in range(1):
            data.append(df[i])

    # 修改为批量循环（原单个循环改为批量循环）
    for i in tqdm(range(0, len(data), batch_size)):
        batch_data = data[i:i + batch_size]  # 取当前批次数据
        texts = []
        # 构建批次输入文本
        for item in batch_data:
            message = [
                {"role": "system", "content": item["instruction"]},
                {"role": "user", "content": item["input"]}
            ]
            text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            texts.append(text)

        # 使用vllm进行批量生成
        outputs = llm.generate(texts, sampling_params)

        for idx, output in enumerate(outputs):
            response = output.outputs[0].text
            item = batch_data[idx]  # 当前样本数据
            example.append({
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"],
                "response": response
            })
        print("已经完成了", (i + batch_size)/len(data))

    final_result = []
    answer_pattern = re.compile(r"####\s*(-?\d+\.?\d*)")
    for i in range(int(len(example)/1)):
        number = int(1*i)
        answer_cal = []
        for j in range(number,number+1):
            try:
                answer = answer_pattern.search(example[j]["response"]).group(1)
                reference = answer_pattern.search(example[j]["output"]).group(1)
                if answer == reference:
                    answer_cal.append(1)
                else:
                    answer_cal.append(0)
            except Exception as e:
                print(f"提取不到信息：{e}")
                answer_cal.append(0)

        result_mean = sum(answer_cal)/1
        result_max = max(answer_cal)
        result_min = min(answer_cal)

        final_result.append([result_mean, result_max, result_min])

    mean_mean = np.mean(final_result, axis=0)[0]
    mean_max = np.mean(final_result, axis=0)[1]
    mean_min = np.mean(final_result, axis=0)[2]

    return example, mean_mean, mean_max, mean_min


def calculate_rouge_l(prediction: str, reference: str) -> dict:
    from rouge_score import rouge_scorer
    # 初始化scorer，指定评估ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)  # use_stemmer：是否词干化（如将过去式转为原型）
    # 计算分数（prediction：生成文本，reference：参考文本）
    scores = scorer.score(reference, prediction)
    # 提取ROUGE-L的分数（包括precision、recall、f1）
    rouge_l = scores['rougeL']
    return rouge_l.fmeasure

from vllm import LLM, SamplingParams

def eval_rougeL(model_name_or_path, test_data_path, batch_size):
    if torch.cuda.device_count() > 1:
        device_count_ = 4
    else:
        device_count_ = 1
    print("显卡一共有{}张，我们使用{}张进行eval".format(torch.cuda.device_count(), device_count_))
    llm = LLM( model=model_name_or_path, tokenizer=model_name_or_path,
               tokenizer_mode="auto",
                tensor_parallel_size=device_count_, gpu_memory_utilization=0.3)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=500, temperature=0.6)
    data = []
    df = json.load(open(test_data_path, "r"))
    for i in range(len(df)):
        for j in range(8):
            data.append(df[i])

    example = []

    # 修改为批量循环（原单个循环改为批量）
    for i in tqdm(range(0, len(data), batch_size)):
        batch_data = data[i:i + batch_size]  # 取当前批次数据
        texts = []
        # 构建批次输入文本
        for item in batch_data:
            message = [
                {"role": "system", "content": item["instruction"]},
                {"role": "user", "content": item["input"]}
            ]
            text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            texts.append(text)

        # 使用vllm进行批量生成
        outputs = llm.generate(texts, sampling_params)

        # 解析批量生成结果
        for idx, output in enumerate(outputs):
            response = output.outputs[0].text
            item = batch_data[idx]  # 当前样本数据
            example.append({
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"],
                "response": response
            })
        print("已经完成了", (i + batch_size)/len(data))

    final_result = []
    for i in tqdm(range(int(len(example) / 8))):
        number = int(8 * i)
        candidates = [example[j]["response"] for j in range(number, number + 8)]
        references = [example[j]["output"] for j in range(number, number + 8)]
        # 并行计算ROUGE-L
        results, stats = calculate_rouge_l_parallel(
            candidates=candidates,
            references=references,
            use_stemmer=True,
            verbose=False,  # 大规模数据时建议关闭详细输出
            max_workers=4  # 指定4个进程
        )
        final_result.append([stats['avg_fmeasure'], stats['max_fmeasure'], stats['min_fmeasure']])

    mean_mean = np.mean(final_result, axis=0)[0]
    mean_max = np.mean(final_result, axis=0)[1]
    mean_min = np.mean(final_result, axis=0)[2]

    return example, mean_mean, mean_max, mean_min



# def eval_rougeL(model_name_or_path, test_data_path, batch_size):
#     model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
#     use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left",
#                                               legacy=False)
#     tokenizer.pad_token_id = 0
#
#     data = []
#     df = json.load(open(test_data_path, "r"))
#     for i in range(len(df)):
#         for j in range(5):
#             data.append(df[i])
#
#     example = []
#     rouge_l_total = 0  # 累计rouge分数
#
#     # 修改为批量循环（原单个循环改为批量）
#     for i in tqdm(range(0, len(data), batch_size)):
#         batch_data = data[i:i + batch_size]  # 取当前批次数据
#         texts = []
#         # 构建批次输入文本
#         for item in batch_data:
#             message = [
#                 {"role": "system", "content": item["instruction"]},
#                 {"role": "user", "content": item["input"]}
#             ]
#             text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
#             texts.append(text)
#
#         # 批量tokenize（原单个处理改为批量）
#         model_inputs = tokenizer(texts, return_tensors='pt', padding=True).to(model.device)  # 新增padding参数
#         generated_ids = model.generate(**model_inputs, max_new_tokens=1000)
#
#         # 解析批量生成结果（原单个解析改为批量）
#         for idx, (input_ids, output_ids) in enumerate(zip(model_inputs.input_ids, generated_ids)):
#             generated_ids_single = output_ids[len(input_ids):]
#             response = tokenizer.decode(generated_ids_single, skip_special_tokens=True)
#             item = batch_data[idx]  # 当前样本数据
#             # 原rouge计算逻辑调整（累计分数）
#             example.append({
#                 "instruction": item["instruction"],
#                 "input": item["input"],
#                 "output": item["output"],
#                 "response": response
#             })
#         print("已经完成了", (i + batch_size)/len(data))
#     final_result = []
#     for i in tqdm(range(int(len(example) / 5))):
#         number = int(5 * i)
#         candidates = [example[j]["response"] for j in range(number, number + 5)]
#         references = [example[j]["output"] for j in range(number, number + 5)]
#         # 并行计算ROUGE-L
#         results, stats = calculate_rouge_l_parallel(
#             candidates=candidates,
#             references=references,
#             use_stemmer=True,
#             verbose=False,  # 大规模数据时建议关闭详细输出
#             max_workers=10  # 指定4个进程
#         )
#         final_result.append([stats['avg_fmeasure'], stats['max_fmeasure'], stats['min_fmeasure']])
#
#     mean_mean = np.mean(final_result, axis=0)[0]
#     mean_max = np.mean(final_result, axis=0)[1]
#     mean_min = np.mean(final_result, axis=0)[2]
#
#     return example, mean_mean, mean_max, mean_min


def write_to_file(file_path, text):
    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(text)
        print(f"已成功将文本写入到 {file_path}")
    except Exception as e:
        print(f"写入文件时出错: {e}")


def save_json(predictions, model_result):
    save_dir = "save_result/{}".format(model_result.split("/")[-2])
    os.makedirs(save_dir, exist_ok=True)  # 添加目录创建逻辑
    # 创建保存目录（如果不存在）
    with open("save_result/{}/{}.json".format(model_result.split("/")[-2], model_result.split("/")[-1]), 'w',
              encoding='utf-8') as file:
        json.dump(predictions, file, ensure_ascii=False, indent=4)


def main():
    args = parse_args()
    if not os.path.exists('eval_result'):
        os.makedirs('eval_result')
    if 'GSM' in args.model_path:
        example, mean_mean, mean_max, mean_min = eval_gsm(args.model_path, "eval_data/gsm8k_test.json", args.batch_size)
        file_path = 'eval_result/gsm.txt'
        sample_text = "{}路径下的模型检测结果为:中间结果为:{}, 最优结果为:{}, 最差结果为:{}\n".format(args.model_path, mean_mean, mean_max, mean_min)
        write_to_file(file_path, sample_text)
        save_json(example, args.model_path)
    elif 'DialogSum' in args.model_path:
        example, mean_mean, mean_max, mean_min = eval_rougeL(args.model_path, "eval_data/DialogSum_test.json", args.batch_size)
        file_path = 'eval_result/DialogSum.txt'
        sample_text = "{}路径下的模型检测结果为:中间结果为:{}, 最优结果为:{}, 最差结果为:{}\n".format(args.model_path, mean_mean, mean_max, mean_min)
        write_to_file(file_path, sample_text)
        save_json(example, args.model_path)
    elif 'BioInstruct' in args.model_path:
        example, mean_mean, mean_max, mean_min = eval_rougeL(args.model_path, "eval_data/BioInstruct_test.json", args.batch_size)
        file_path = 'eval_result/BioInstruct.txt'
        sample_text = "{}路径下的模型检测结果为:中间结果为:{}, 最优结果为:{}, 最差结果为:{}\n".format(args.model_path, mean_mean, mean_max, mean_min)
        write_to_file(file_path, sample_text)
        save_json(example, args.model_path)

    elif 'Math' in args.model_path:
        example, mean_mean, mean_max, mean_min = eval_math(args.model_path, "eval_data/math_test.json", args.batch_size)
        file_path = 'eval_result/math.txt'
        sample_text = "{}路径下的模型检测结果为:中间结果为:{}, 最优结果为:{}, 最差结果为:{}\n".format(args.model_path,
                                                                                                      mean_mean,
                                                                                                      mean_max,
                                                                                                      mean_min)
        write_to_file(file_path, sample_text)
        save_json(example, args.model_path)
    else:
        example, mean_mean, mean_max, mean_min = eval_rougeL(args.model_path, "eval_data/BioInstruct_test.json", args.batch_size)
        file_path = 'eval_result/laji.txt'
        sample_text = "{}路径下的模型检测结果为:中间结果为:{}, 最优结果为:{}, 最差结果为:{}\n".format(args.model_path, mean_mean, mean_max, mean_min)
        write_to_file(file_path, sample_text)
        save_json(example, args.model_path)


if __name__ == "__main__":
    main()