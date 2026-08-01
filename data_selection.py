import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import os
import argparse
import pickle


class LlamaFeatureExtractor:
    def __init__(self, model_path, batch_size=32):
        # 加载模型和分词器，保持原有参数
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # 尝试使用fast tokenizer加速（若结果一致可保留）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,  # 改为fast tokenizer加速分词
            padding_side="left",
            legacy=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 存储钩子输出（保留在GPU上以减少传输）
        self.gate_outputs = []
        self.up_outputs = []
        self.down_outputs = []

        # 注册钩子
        self._register_hooks()
        self.batch_size = batch_size  # 批量处理大小

    def obtain_parm(self):
        return sum(p.numel() for p in self.model.parameters())

    def _register_hooks(self):
        # 钩子函数不转移数据到CPU，保留在GPU
        def gate_hook(module, input, output):
            self.gate_outputs.append(output.detach())  # 仅detach，不转CPU

        def up_hook(module, input, output):
            self.up_outputs.append(output.detach())

        def down_hook(module, input, output):
            self.down_outputs.append(output.detach())

        # 注册所有层的钩子
        for layer in self.model.model.layers:
            layer.mlp.gate_proj.register_forward_hook(gate_hook)
            layer.mlp.up_proj.register_forward_hook(up_hook)
            layer.mlp.down_proj.register_forward_hook(down_hook)

    def extract_features(self, input_texts):
        # 清空之前的结果
        self.gate_outputs.clear()
        self.up_outputs.clear()
        self.down_outputs.clear()

        # 批量处理chat template
        texts = [self.tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=False
        ) for msg in input_texts]

        # 批量分词（添加truncation避免长文本报错，保持与单样本行为一致）
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            # padding=True,
            truncation=True,  # 确保长文本处理一致
            # padding_side=self.tokenizer.padding_side
        ).to(self.model.device)

        # 前向传播（保持无梯度）
        with torch.no_grad():
            self.model(**inputs)

        # 返回GPU上的张量列表（[num_layers, batch_size, seq_len, hidden_size]）
        return self.gate_outputs, self.up_outputs, self.down_outputs

    def count_mlp_parameters(self):
        total_params = 0
        mlp_params_list = []

        # 遍历每一层 decoder block
        for i, layer in enumerate(self.model.model.layers):
            # 每层 MLP 模块
            mlp = layer.mlp

            # 统计当前 MLP 层的参数总数
            layer_params = sum(p.numel() for p in mlp.parameters())
            total_params += layer_params

            mlp_params_list.append((layer_params))

        return sum(p.numel() for p in self.model.parameters()), sum(mlp_params_list)


from collections import defaultdict

import gc
from dataclasses import dataclass
from typing import List, Tuple, Set

import numpy as np
import torch
from tqdm import tqdm


def _flatten_last_token(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    将特征列表堆叠，并提取最后一个 token 的激活。
    返回 shape: (num_layers, hidden_dim_merged) 的 2D 张量。
    """
    stacked = torch.stack(tensors)  # (num_layers, ..., seq_len, hidden)
    last_token = stacked[..., -1, :]  # 仅保留最后一个 token
    return last_token.reshape(last_token.size(0), -1)  # 合并除层以外的维度


@dataclass
class TensorRunningStats:
    """
    使用 Welford 算法的向量化实现来维护整个 (num_layers, hidden_dim) 矩阵的统计量。
    """
    count: np.ndarray | None = None
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def _maybe_initialize(self, shape: Tuple[int, int]) -> None:
        if self.count is None:
            self.count = np.zeros(shape, dtype=np.int32)
            self.mean = np.zeros(shape, dtype=np.float64)
            self.m2 = np.zeros(shape, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        """
        values: shape (num_layers, hidden_dim) 的 numpy 数组。
        """
        if values.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {values.shape}")

        self._maybe_initialize(values.shape)

        # 逐元素向量化的 Welford 更新
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        delta2 = values - self.mean
        self.m2 += delta * delta2

    def collect_indices(
            self,
            min_mean: float,
            relative_variance_threshold: float,
    ) -> Set[Tuple[int, int]]:
        if self.count is None:
            return set()

        # 只有出现过至少 2 次的神经元才有方差
        variance = np.zeros_like(self.mean)
        valid = self.count > 1
        variance[valid] = self.m2[valid] / (self.count[valid] - 1)

        # 计算相对方差：variance / mean^2
        rel_var = np.full_like(self.mean, np.inf)
        positive_mean = self.mean > 0.0
        rel_var[positive_mean] = variance[positive_mean] / (
                self.mean[positive_mean] ** 2 + 1e-12
        )

        mask = positive_mean & (rel_var < relative_variance_threshold)
        if min_mean > 0.0:
            mask &= self.mean >= min_mean

        indices = np.argwhere(mask)
        return {tuple(map(int, idx)) for idx in indices}


def get_activated_neuron_positions_union(
        extractor: "LlamaFeatureExtractor",
        critical_data: List[dict],
        neuron_ratio: float,
        *,
        batch_size: int = 256,
        relative_variance_threshold: float = 0.10,
        log_progress: bool = True,
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    处理流程：
      1. 按批遍历 critical_data；
      2. 对每条数据提取最后一个 token 的 gate/up/down 激活；
      3. 用向量化的 Welford 统计更新；
      4. 每批结束进行垃圾回收，降低峰值内存；
      5. 汇总时保留平均激活值 > 0、均值阈值 >= neuron_ratio、相对方差 < relative_variance_threshold 的神经元；
      6. 返回符合条件的 gate/up/down 神经元索引集合。
    """
    final_threshold = max(float(neuron_ratio), 0.0)
    total = len(critical_data)
    num_batches = (total + batch_size - 1) // batch_size

    gate_stats = TensorRunningStats()
    up_stats = TensorRunningStats()
    down_stats = TensorRunningStats()

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = critical_data[start:end]

        iterable = tqdm(batch, desc=f"Processing batch {batch_idx + 1}/{num_batches}") if log_progress else batch

        for item in iterable:
            message = [[
                {"role": "system", "content": item["instruction"]},
                {"role": "user", "content": item["input"]},
            ]]

            with torch.no_grad():
                gate, up, down = extractor.extract_features(message)

            gate_np = _flatten_last_token(gate).detach().cpu().numpy().astype(np.float32, copy=False)
            up_np = _flatten_last_token(up).detach().cpu().numpy().astype(np.float32, copy=False)
            down_np = _flatten_last_token(down).detach().cpu().numpy().astype(np.float32, copy=False)

            gate_stats.update(gate_np)
            up_stats.update(up_np)
            down_stats.update(down_np)

            # 如需日志，可在此处按需抽样记录少量信息，而不是全量保存
            # item["gate_activation_logs"] = ...
            # item["up_activation_logs"] = ...
            # item["down_activation_logs"] = ...

            del message, gate, up, down
            del gate_np, up_np, down_np
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del batch
        gc.collect()

    gate_union = gate_stats.collect_indices(final_threshold, relative_variance_threshold)
    up_union = up_stats.collect_indices(final_threshold, relative_variance_threshold)
    down_union = down_stats.collect_indices(final_threshold, relative_variance_threshold)

    return gate_union, up_union, down_union


def format_positions_to_mask(positions_set: set, num_layers: int) -> list:
    """
    将 (层, 神经元) 索引集合转换为按层组织的列表掩码。

    Args:
        positions_set (set): 包含 (layer_index, neuron_index) 元组的集合。
        num_layers (int): 模型的总层数。

    Returns:
        list: 一个列表，其中每个子列表包含该层被激活的神经元索引。
              格式: [[layer_0_indices], [layer_1_indices], ...]
    """
    # 初始化一个长度为 num_layers 的空列表嵌套列表
    mask = [[] for _ in range(num_layers)]

    # 遍历集合中的每一个 (层, 神经元) 对
    for layer_idx, neuron_idx in positions_set:
        if layer_idx < num_layers:
            # 将神经元索引添加到对应层的子列表中
            mask[layer_idx].append(neuron_idx)

    # （可选）对每个子列表中的神经元索引进行排序，使其更整洁
    for layer_indices in mask:
        layer_indices.sort()

    return mask


from typing import Iterable, Dict, Any, List, Tuple, Set
import numpy as np
import torch
from tqdm import tqdm


def _prepare_index_arrays(indices: Set[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    将神经元索引集合转换为两个 numpy 数组 (layers, neurons)，方便进行向量化取值。
    如果集合为空，则返回空数组。
    """
    if not indices:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    layered = np.array(list(indices), dtype=np.int64)
    return layered[:, 0], layered[:, 1]


from typing import Iterable, Dict, Any, List, Tuple, Set, Optional
import math

import numpy as np
import torch
from tqdm import tqdm


def _prepare_index_arrays(indices: Set[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    将神经元索引集合转换为两个 numpy 数组 (layers, neurons)，方便进行向量化取值。
    如果集合为空，则返回空数组。
    """
    if not indices:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    layered = np.array(list(indices), dtype=np.int64)
    return layered[:, 0], layered[:, 1]


def _mean_for_selected(
        matrix: np.ndarray,
        layers: np.ndarray,
        neurons: np.ndarray,
) -> Optional[float]:
    """
    在给定的层/神经元索引上求平均。若索引集合为空，返回 None。
    """
    if layers.size == 0:
        return None
    values = matrix[layers, neurons]
    return float(values.mean())


def rank_samples_by_selected_mean_activation(
        extractor: "LlamaFeatureExtractor",
        critical_data: List[dict],
        gate_indices: Set[Tuple[int, int]],
        up_indices: Set[Tuple[int, int]],
        down_indices: Set[Tuple[int, int]],
        *,
        batch_size: int = 32,
        log_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    计算每条样本在已筛选神经元集合上的平均激活值，并按总平均值降序排序。

    参数：
        extractor: 你的特征抽取器，需提供 extract_features(message) -> (gate, up, down)。
        critical_data: 输入数据列表，元素应含 "instruction" 与 "input" 字段。
        gate_indices/up_indices/down_indices: 已筛选出的神经元集合。
        batch_size: 批处理大小。
        log_progress: 是否显示 tqdm 进度条。

    返回：
        列表元素示例：
        {
            "sample_index": 12,
            "sample_id": "abc-123",        # 若 critical_data 中有 "id"
            "gate_mean": 0.83,
            "up_mean": 0.91,
            "down_mean": 0.67,
            "total_mean": 0.80,            # 三者的平均（忽略 None）
        }
        列表已按 total_mean 降序排序。
    """
    total = len(critical_data)
    num_batches = (total + batch_size - 1) // batch_size

    gate_layers, gate_neurons = _prepare_index_arrays(gate_indices)
    up_layers, up_neurons = _prepare_index_arrays(up_indices)
    down_layers, down_neurons = _prepare_index_arrays(down_indices)

    results: List[Dict[str, Any]] = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = critical_data[start:end]

        iterable: Iterable[Tuple[int, dict]] = tqdm(
            enumerate(batch, start=start),
            desc=f"Mean coverage batch {batch_idx + 1}/{num_batches}",
        ) if log_progress else enumerate(batch, start=start)

        for global_idx, item in iterable:
            message = [[
                {"role": "system", "content": item["instruction"]},
                {"role": "user", "content": item["input"]},
            ]]
            with torch.no_grad():
                gate, up, down = extractor.extract_features(message)

            gate_np = _flatten_last_token(gate).detach().cpu().numpy().astype(np.float32, copy=False)
            up_np = _flatten_last_token(up).detach().cpu().numpy().astype(np.float32, copy=False)
            down_np = _flatten_last_token(down).detach().cpu().numpy().astype(np.float32, copy=False)

            gate_mean = _mean_for_selected(gate_np, gate_layers, gate_neurons)
            up_mean = _mean_for_selected(up_np, up_layers, up_neurons)
            down_mean = _mean_for_selected(down_np, down_layers, down_neurons)

            # 合并三个子均值（忽略 None），得到总平均
            valid_means = [x for x in (gate_mean, up_mean, down_mean) if x is not None]
            total_mean = float(np.mean(valid_means)) if valid_means else None

            results.append({
                "sample_index": global_idx,
                "sample_id": item.get("id"),
                "gate_mean": gate_mean,
                "up_mean": up_mean,
                "down_mean": down_mean,
                "total_mean": total_mean,
            })

            del message, gate, up, down
            del gate_np, up_np, down_np
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del batch
        gc.collect()

    # 处理排序时 None 的情况：将 None 当作负无穷
    def sort_key(row: Dict[str, Any]) -> float:
        value = row.get("total_mean")
        return value if value is not None else float("-inf")

    results.sort(key=sort_key, reverse=True)
    return results


import math
from typing import List, Dict, Any, Iterable


def select_top_percentiles_sources(
        sorted_results: List[Dict[str, Any]],
        critical_data: List[dict],
        percent_list: Iterable[float] = (0.01, 0.05, 0.10),
        *,
        min_items: int = 1,
        deep_copy: bool = False,
) -> Dict[float, List[dict]]:
    """
    根据降序排列的 sorted_results，从 critical_data 中抽取指定比例的原始数据条目。

    参数：
        sorted_results: rank_samples_by_selected_mean_activation 的返回值（需已按 total_mean 降序）。
        critical_data: 原始数据列表。
        percent_list: 要抽取的比例集合，例如 (0.01, 0.05, 0.10)。
        min_items: 防止比例过小导致 0 条，至少返回的条数。
        deep_copy: 是否对返回的数据执行 deepcopy，避免外部修改影响原始数据。

    返回：
        dict，键为比例，值为对应的源数据列表。
    """
    total = len(sorted_results)
    if total == 0:
        return {p: [] for p in percent_list}

    output: Dict[float, List[dict]] = {}

    for p in percent_list:
        if p <= 0:
            output[p] = []
            continue
        # 目标选取数量
        count = max(min_items, math.ceil(total * p))
        count = min(count, total)
        # 计算中间段起止位置（尽量居中）
        start = (total - count) // 2
        end = start + count
        if end > total:
            end = total
            start = end - count
        subset = []
        for entry in sorted_results[start:end]:
            idx = entry["sample_index"]
            source_item = critical_data[idx]
            if deep_copy:
                source_item = copy.deepcopy(source_item)
            subset.append(source_item)
        output[p] = subset
    return output


def ensure_directory(path):
    if path:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--neuron_ratio", type=float, default=0)
    return parser.parse_args()


def main():
    """主函数：从关键数据中找到激活神经元位置并集，然后格式化并保存为pkl文件。"""
    args = parse_args()
    # 加载关键数据
    critical_data = json.load(open(args.train_data_path, "r"))
    # 初始化模型和特征提取器
    print(f"Loading model from: {args.model_path}")
    extractor = LlamaFeatureExtractor(args.model_path)

    # 获取位置并集
    gate_positions, up_positions, down_positions = get_activated_neuron_positions_union(extractor, critical_data,
                                                                                        args.neuron_ratio)

    print("\n--- Position Union Calculation Complete ---")
    print(f"Total unique activated positions in gate_proj: {len(gate_positions)}")
    print(f"Total unique activated positions in up_proj: {len(up_positions)}")
    print(f"Total unique activated positions in down_proj: {len(down_positions)}")

    # --- 新增步骤：格式化并保存 ---
    print("\nFormatting positions into layer-wise masks...")

    # 从加载的模型中动态获取层数，这是最可靠的方法
    num_layers = len(extractor.model.model.layers)
    print(f"Detected {num_layers} layers in the model.")

    # 将位置集合格式化为掩码列表
    gate_mask = format_positions_to_mask(gate_positions, num_layers)
    up_mask = format_positions_to_mask(up_positions, num_layers)
    down_mask = format_positions_to_mask(down_positions, num_layers)

    important_neuron = []
    important_neuron.append(gate_mask)
    important_neuron.append(up_mask)
    important_neuron.append(down_mask)

    def store_neuron(train_data_path, model_path, important_neuron):
        if "gsm" in train_data_path:
            save_name = "neuron_weight/GSM_{}.pkl".format(model_path.split("/")[-1])
        elif "DialogSum" in train_data_path:
            save_name = "neuron_weight/DialogSum_{}.pkl".format(model_path.split("/")[-1])
        else:
            save_name = "neuron_weight/Law_{}.pkl".format(model_path.split("/")[-1])
        with open(save_name, 'wb') as f:
            pickle.dump(important_neuron, f)

    store_neuron(args.train_data_path, args.model_path, important_neuron)
    print("The file contains a dictionary with keys: 'gate_mask', 'up_mask', 'down_mask'.")

    rank_data = rank_samples_by_selected_mean_activation(
        extractor,
        critical_data,
        gate_positions,
        up_positions,
        down_positions,
        batch_size=16,
    )

    # mean_report 已经按照 total_mean 降序排序，无需再次排序
    top_slices = select_top_percentiles_sources(
        rank_data,
        critical_data,
        percent_list=(0.01, 0.05, 0.10),
    )

    top_1_percent = top_slices[0.01]
    top_5_percent = top_slices[0.05]
    top_10_percent = top_slices[0.10]

    # 生成保存路径
    if "gsm" in args.train_data_path:
        save_name_1 = f"data/GSM_{args.model_path.split('/')[-1]}_0.01.json"
        save_name_2 = f"data/GSM_{args.model_path.split('/')[-1]}_0.05.json"
        save_name_3 = f"data/GSM_{args.model_path.split('/')[-1]}_0.10.json"
    elif "DialogSum" in args.train_data_path:
        save_name_1 = f"data/DialogSum_{args.model_path.split('/')[-1]}_0.01.json"
        save_name_2 = f"data/DialogSum_{args.model_path.split('/')[-1]}_0.05.json"
        save_name_3 = f"data/DialogSum_{args.model_path.split('/')[-1]}_0.10.json"
    else:
        save_name_1 = f"data/Law_{args.model_path.split('/')[-1]}_0.01.json"
        save_name_2 = f"data/Law_{args.model_path.split('/')[-1]}_0.05.json"
        save_name_3 = f"data/Law_{args.model_path.split('/')[-1]}_0.10.json"
    # 确保目录存在
    ensure_directory(save_name_1)
    ensure_directory(save_name_2)
    ensure_directory(save_name_3)

    with open(save_name_1, 'w', encoding='utf-8') as file:
        json.dump(top_1_percent, file, ensure_ascii=False, indent=4)
    with open(save_name_2, 'w', encoding='utf-8') as file:
        json.dump(top_5_percent, file, ensure_ascii=False, indent=4)
    with open(save_name_3, 'w', encoding='utf-8') as file:
        json.dump(top_10_percent, file, ensure_ascii=False, indent=4)

    print("保存成功!!!")


if __name__ == "__main__":
    # 运行新任务
    main()