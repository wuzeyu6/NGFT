import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import os

import pickle
import argparse
import numpy as np
import math
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy.signal import correlate

def generate_data(model, model_path, data, batch_size):
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast_tokenizer, padding_side="left",
                                              legacy=False)
    tokenizer.pad_token_id = 0
    example = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch_data = data[i:i + batch_size]  # 取当前批次数据
        texts = []
        for item in batch_data:
            if "Qwen" in model_path:
                texts.append(format_qwen_template(item, "no", ""))
            elif "Llama" in model_path:
                texts.append(format_llama3_template(item, "no", ""))
            else:
                texts.append(format_mistral_template(item, "no", ""))

    model_inputs = tokenizer(texts, return_tensors='pt', padding=True).to(model.device)  # 新增padding参数
    generated_ids = model.generate(**model_inputs, max_new_tokens=1000)
    # 解析批量生成结果（原单个解析改为批量）
    for idx, (input_ids, output_ids) in enumerate(zip(model_inputs.input_ids, generated_ids)):
        generated_ids_single = output_ids[len(input_ids):]
        response = tokenizer.decode(generated_ids_single, skip_special_tokens=True)
        item = batch_data[idx]  # 当前样本数据
        example.append(format_qwen_template(item, "fine_tune", response))
    return example

def format_qwen_template(data: dict, type_, answer) -> str:
    """
    Formats a dictionary with 'instruction', 'input', and 'output' keys
    into the Qwen chat format.

    Args:
        data: A dictionary containing the keys 'instruction', 'input', and 'output'.

    Returns:
        A string formatted according to the Qwen template.


    Type: best为最佳答案；fine_tune为微调答案；no为不需要答案
    """
    instruction = data.get("instruction", "")
    user_input = data.get("input", "")
    if type_ == "best":
        assistant_output = data.get("output", "")
    elif type_ == "fine_tune":
        assistant_output = answer

    # Build the system part
    system_part = f"<|im_start|>system\n{instruction}<|im_end|>\n" if instruction else ""

    # Build the user part
    user_part = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

    if type_ != "no":
        # Build the assistant part
        assistant_part = f"{assistant_output}<|im_end|>"

        # Combine all parts
        # The separator between system and user is already included in system_part
        # The separator between user and assistant is already included in user_part
        full_text = f"{system_part}{user_part}{assistant_part}"
    else:
        full_text = f"{system_part}{user_part}"

    return full_text


def format_llama3_template(data: dict, type_, answer, bos_token: str = "<|begin_of_text|>") -> str:
    """
    Formats a dictionary with 'instruction', 'input', and 'output' keys
    into the Llama 3 chat format.

    Args:
        data: A dictionary containing the keys 'instruction', 'input', and 'output'.
        bos_token: The beginning of text token. For Llama 3, this is typically '<|begin_of_text|>'.

    Returns:
        A string formatted according to the Llama 3 template.
    """
    instruction = data.get("instruction", "")
    user_input = data.get("input", "")
    if type_ == "best":
        assistant_output = data.get("output", "")
    elif type_ == "fine_tune":
        assistant_output = answer

    # The template starts with a Beginning-Of-Sentence (BOS) token.
    text_parts = [bos_token]

    # Add the system message if an instruction is provided.
    if instruction:
        text_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>")

    # Add the user message.
    text_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>")

    if type_ != "no":
        # Add the assistant's response, starting with the assistant header.
        text_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_output}<|eot_id|>")

    # Join all parts. The template doesn't specify a separator between messages,
    # as each message block is self-contained with a header and an end token.
    return "".join(text_parts)

def format_mistral_template(data: dict, type_, answer, bos_token: str = "<s>", eos_token: str = "</s>") -> str:
    """
    Formats a dictionary with 'instruction', 'input', and 'output' keys
    into the Mistral Instruct chat format.

    Args:
        data: A dictionary containing the keys 'instruction', 'input', and 'output'.
        bos_token: The beginning of sequence token.
        eos_token: The end of sequence token.

    Returns:
        A string formatted according to the Mistral template.
    """
    instruction = data.get("instruction", "")
    user_input = data.get("input", "")
    if type_ == "best":
        assistant_output = data.get("output", "")
    elif type_ == "fine_tune":
        assistant_output = answer

    text_parts = [bos_token]

    # The system prompt is placed before the user prompt.
    # The template defines it as "{{content}}\n\n".
    if instruction:
        text_parts.append(f"{instruction}\n\n")

    # The user input is wrapped in [INST] tags.
    text_parts.append(f"[INST]{user_input}[/INST]")

    if type_ != "no":
        # The assistant output follows immediately.
        text_parts.append(assistant_output)
    # The entire sequence is terminated with an EOS token for training.
    text_parts.append(eos_token)

    return "".join(text_parts)

class LlamaFeatureExtractor:
    def __init__(self, model_path, model):
        # Load model and tokenizer
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.float16,
        #     device_map="auto"
        # )
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            padding_side="left",
            legacy=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize storage for hooks
        self.gate_outputs = []
        self.up_outputs = []
        self.down_outputs = []

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        # Define hook functions with proper tensor detachment
        def gate_hook(module, input, output):
            # 保存时增加一个维度来区分不同批次
            self.gate_outputs.append(output.detach().cpu().unsqueeze(0))

        def up_hook(module, input, output):
            self.up_outputs.append(output.detach().cpu().unsqueeze(0))

        def down_hook(module, input, output):
            self.down_outputs.append(output.detach().cpu().unsqueeze(0))

        # Register hooks to all layers
        for layer in self.model.model.layers:
            layer.mlp.gate_proj.register_forward_hook(gate_hook)
            layer.mlp.up_proj.register_forward_hook(up_hook)
            layer.mlp.down_proj.register_forward_hook(down_hook)

    def extract_features(self, input_texts, max_length=1000):
        """
        处理批量输入文本并提取特征

        Args:
            input_texts: 文本列表，每个元素是一个对话或文本
            max_length: 最大序列长度，超过将被截断

        Returns:
            三个numpy数组，分别包含gate、up和down的输出特征
            形状为 (num_layers, batch_size, seq_len, hidden_size)
        """
        # Clear previous results
        self.gate_outputs.clear()
        self.up_outputs.clear()
        self.down_outputs.clear()

        # 处理批量文本
        processed_texts = input_texts
        # 批量tokenize，自动处理padding和截断
        inputs = self.tokenizer(
            processed_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.model.device)

        # Forward pass
        with torch.no_grad():
            self.model(**inputs)

        # 合并所有层的输出并调整维度顺序
        # 结果形状: (num_layers, batch_size, seq_len, hidden_size)
        gate_features = torch.cat(self.gate_outputs, dim=0).to(torch.float32).numpy()
        up_features = torch.cat(self.up_outputs, dim=0).to(torch.float32).numpy()
        down_features = torch.cat(self.down_outputs, dim=0).to(torch.float32).numpy()

        return gate_features, up_features, down_features


def matrix_calculate(model_path, extractor, data, batch):
    gate = []
    up = []
    down = []
    if len(data) % batch != 0:
        for s in tqdm(range(int(len(data) / batch)+1)):
            if s * batch + batch > len(data):
                gate_, up_, down_ = extractor.extract_features(data[s*batch:])
            else:
                gate_, up_, down_ = extractor.extract_features(data[s*batch:s*batch+batch])
            gate_ = np.mean(gate_, axis=2)
            up_ = np.mean(up_, axis=2)
            down_ = np.mean(down_, axis=2)
            for i in range(gate_.shape[1]):
                gate.append(gate_[:,i,:])
                up.append(up_[:,i,:])
                down.append(down_[:,i,:])
    else:
        for s in tqdm(range(int(len(data)/batch))):
            gate_, up_, down_ = extractor.extract_features(data[s*batch:s*batch+batch])
            gate_ = np.mean(gate_, axis=2)
            up_ = np.mean(up_, axis=2)
            down_ = np.mean(down_, axis=2)
            for i in range(batch):
                gate.append(gate_[:,i,:])
                up.append(up_[:,i,:])
                down.append(down_[:,i,:])
    return np.array(gate), np.array(up), np.array(down)


def element_wise_similarity(arr1, arr2, similarity_func):
    """
    对两个数组的对应元素应用相似度函数，返回平均相似度
    忽略无法计算相似度的位置
    """
    if len(arr1) != len(arr2):
        raise ValueError("两个数组的元素数量必须相同")

    def count_greater_than_zero(lst):
        count = 0
        for num in lst:
            if num > 0:
                count += 1
        return count

    similarities = []
    for a, b in zip(arr1, arr2):
        try:
            # 确保元素是可迭代的
            if not hasattr(a, '__iter__') or not hasattr(b, '__iter__'):
                continue

            sim = similarity_func(a, b)
            if not math.isnan(sim):
                similarities.append(sim)
        except:
            continue

    if not similarities:
        return math.nan

    sim = count_greater_than_zero(similarities)
    return sum(similarities) / sim

# def overlapping_cosine_similarity(a, b):
#     assert a.dim() == b.dim(), f"输入a和b的维度需一致，当前a维度={a.dim()}, b维度={b.dim()}"
#     if a.dim() == 2:
#         assert a.shape[0] == b.shape[0], f"2D输入时batch_size需一致，当前a batch={a.shape[0]}, b batch={b.shape[0]}"
#     # 2. 计算最小长度（1D取序列长度，2D取每批序列的最小长度）
#     if a.dim() == 1:
#         len_a, len_b = a.shape[0], b.shape[0]
#     else:  # 2D: [batch_size, seq_len]，按批次计算每个样本的min_len
#         len_a, len_b = a.shape[1], b.shape[1]
#     min_len = min(len_a, len_b)
#     # 3. 处理空序列（min_len=0），返回NaN（PyTorch中用torch.nan）
#     if min_len == 0:
#         return torch.nan.to(a.device)  # 确保NaN与输入在同一设备（如GPU）
#
#     # 4. 截断到最小长度（重叠部分）
#     a_truncated = a[..., :min_len]  # ... 适配1D/2D：1D→a[:min_len]，2D→a[:, :min_len]
#     b_truncated = b[..., :min_len]
#     # 5. 计算余弦相似度：cos_sim = (a·b) / (||a|| * ||b||)
#     # 点积：1D用torch.dot，2D用torch.bmm（批量矩阵乘法，需扩展维度）
#     if a.dim() == 1:
#         dot_product = torch.dot(a_truncated, b_truncated)
#     else:
#         # 2D时转为 [batch_size, 1, min_len] 和 [batch_size, min_len, 1]，点积后 squeeze 为 [batch_size]
#         dot_product = torch.bmm(a_truncated.unsqueeze(1), b_truncated.unsqueeze(2)).squeeze(-1).squeeze(-1)
#     # 计算L2模长（避免除以0，加eps）
#     norm_a = torch.norm(a_truncated, p=2, dim=-1)  # dim=-1：按最后一维（序列维度）计算模长
#     norm_b = torch.norm(b_truncated, p=2, dim=-1)
#     norm_product = norm_a * norm_b + eps  # 加eps防止数值错误
#     # 最终余弦相似度
#     cos_sim = dot_product / norm_product
#
#     return cos_sim.item()


def overlapping_cosine_similarity(a, b):
    """计算两个不同长度序列的重叠部分的余弦相似度"""
    # mse1 = mse_loss(a, b)
    #
    # for j in range(min_len):
    #     if j == 0:
    #         result = mse_loss(a[j], b[j])
    #     if j > 0:
    #         result = result + mse_loss(a[j], b[j])
    # return result

    import torch.nn.functional as F
    mse_loss = torch.nn.MSELoss()
    mse1 = mse_loss(a.to(torch.float32), b.to(torch.float32))
    # diff_mask = (a != b)
    # elements_a_diff = a[diff_mask]
    # elements_b_diff = b[diff_mask]
    # mse2 = mse_loss(elements_b_diff, elements_a_diff)
    return mse1


import torch
import torch.nn.functional as F


def kl_similarity(p, q, dist_dim=-1, reduction='batchmean'):
    """
    计算三维矩阵的KL散度（D_KL(P||Q)），其中第二个维度是batch维度（dim=1）

    参数:
        p: 三维张量，形状为 (dim1, batch_size, dim2)，代表分布P
        q: 三维张量，形状与p一致，代表分布Q
        dist_dim: 分布维度（计算KL散度的维度，如特征维度），默认最后一维(-1)
        reduction: 聚合方式：
            - 'sum': 所有元素求和
            - 'mean': 所有元素取平均
            - 'batchmean': 先对batch维度（dim=1）取平均，再对其他维度取平均（推荐）
    返回:
        kl: 标量，KL散度结果（保留梯度）
    """
    # 数值稳定性：避免log(0)或负数
    epsilon = 1e-8
    p = p + epsilon
    q = q + epsilon

    # # （可选）再次校验（确认修正有效，可保留用于调试）
    # if (p <= 0).any():
    #     # 若仍有非正值，打印具体位置和值（辅助定位异常）
    #     neg_mask = (p <= 0)
    #     print("p中仍有非正值，位置：", torch.where(neg_mask))
    #     print("p中非正值：", p[neg_mask])
    #     raise ValueError("修正后p仍包含非正值，可能输入有极端异常值")
    # if (q <= 0).any():
    #     neg_mask = (q <= 0)
    #     print("q中仍有非正值，位置：", torch.where(neg_mask))
    #     print("q中非正值：", q[neg_mask])
    #     raise ValueError("修正后q仍包含非正值，可能输入有极端异常值")

    # 归一化：在分布维度上确保和为1（概率分布要求）
    p_sum = p.sum(dim=dist_dim, keepdim=True)
    q_sum = q.sum(dim=dist_dim, keepdim=True)
    # 避免除以零（若总和接近0，强制设为极小值）
    p_sum = torch.clamp(p_sum, min=1e-12)
    q_sum = torch.clamp(q_sum, min=1e-12)
    p = p / p_sum
    q = q / q_sum

    # 逐元素计算KL散度：P * log(P/Q)
    element_kl = p * (torch.log(p) - torch.log(q))

    # 在分布维度上求和，得到每个(non-dist, batch)位置的KL值
    # 例如：若输入形状为(time_steps, batch, features)，dist_dim=2，则结果形状为(time_steps, batch)
    kl_per_group = element_kl.sum(dim=dist_dim)

    # 根据reduction聚合结果（重点处理batch维度dim=1）
    if reduction == 'sum':
        return kl_per_group.sum()
    elif reduction == 'mean':
        return kl_per_group.mean()
    elif reduction == 'batchmean':
        # 先对batch维度（dim=1）取平均，再对剩余维度（如time_steps）取平均
        return kl_per_group.mean(dim=1).mean()  # dim=1是batch维度
    else:
        raise ValueError(f"不支持的reduction方式: {reduction}，可选'sum'/'mean'/'batchmean'")


def overlapping_pearson_similarity(a, b):
    """计算两个不同长度序列的重叠部分的皮尔逊相关系数"""
    min_len = min(len(a), len(b))
    if min_len < 2:  # 皮尔逊相关需要至少2个数据点
        return math.nan

    a_truncated = a[:min_len]
    b_truncated = b[:min_len]

    corr, _ = pearsonr(a_truncated, b_truncated)
    return corr


# def dtw_similarity(a, b):
#     """使用动态时间规整计算两个不同长度序列的相似度"""
#     if len(a) == 0 or len(b) == 0:
#         return math.nan
#
#     # 计算DTW距离
#     distance, _, _, _ = dtw(a, b, dist=lambda x, y: abs(x - y))
#
#     # 归一化距离为相似度 (值越大越相似)
#     max_possible = max(max(a) - min(a), max(b) - min(b)) if a and b else 1
#     return 1 / (1 + distance / max_possible)


def cross_correlation_similarity(a, b):
    """使用互相关计算两个不同长度序列的相似度"""
    if len(a) == 0 or len(b) == 0:
        return math.nan

    # 计算互相关
    corr = correlate(a, b, mode='valid')

    # 归一化
    norm = math.sqrt(sum(x ** 2 for x in a) * sum(x ** 2 for x in b))
    if norm == 0:
        return math.nan

    return max(corr) / norm


def analyze_irregular_array_similarity(arr1, arr2):
    """
    全面分析两个不规则数组的相似度
    arr1和arr2是由不同长度子数组组成的数组
    """
    if len(arr1) != len(arr2):
        raise ValueError("两个数组的元素数量必须相同")

    results = {
        "重叠余弦相似度平均值": element_wise_similarity(
            arr1, arr2, overlapping_cosine_similarity),
        "重叠皮尔逊相关系数平均值": element_wise_similarity(
            arr1, arr2, overlapping_pearson_similarity),
        # "DTW相似度平均值": element_wise_similarity(
        #     arr1, arr2, dtw_similarity),
        "互相关相似度平均值": element_wise_similarity(
            arr1, arr2, cross_correlation_similarity)
    }

    return results