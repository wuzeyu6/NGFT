import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from tqdm import tqdm
import warnings
from collections import defaultdict

# 导入rouge_scorer
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("错误: 请先安装 rouge-score 库。运行 'pip install rouge-score'")
    exit()

# 忽略不影响结果的警告
warnings.filterwarnings("ignore", category=UserWarning)


class NeuronAnalyzer:
    def __init__(self,
                 model_name="Qwen/Qwen2.5-0.5B-Instruct",
                 device=None,
                 seed=42,
                 num_layers_to_track=8,
                 top_k_per_layer=1024,
                 activation_cap=1.5):
        """初始化神经元分析器，只研究输入最后一个token的神经元，并过滤掉激活值 > activation_cap 的神经元"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.seed = seed
        self.num_layers_to_track = num_layers_to_track
        self.top_k_per_layer = top_k_per_layer
        self.activation_cap = activation_cap

        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        print(f"正在从Hugging Face加载模型: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            temperature=0,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        ).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'left'

        print(f"模型和分词器加载完成，运行在设备: {self.device}")

        self.activations = {}
        self.ffn_module_names = []
        self._register_hooks()

        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def _register_hooks(self):
        """为模型中选定的前馈网络模块注册钩子（仅保存最后一个token的激活）"""
        target_module_suffix = "mlp.up_proj"
        selected = []

        for name, module in self.model.named_modules():
            if name.endswith(target_module_suffix):
                selected.append((name, module))

        if self.num_layers_to_track is not None:
            selected = selected[-self.num_layers_to_track:]

        for name, module in selected:
            self.ffn_module_names.append(name)

            def make_hook(module_name):
                def hook_fn(module, input, output):
                    act = output[0] if isinstance(output, tuple) else output
                    # 只记录最后一个token的激活: shape [batch, hidden]
                    self.activations[module_name] = act[:, -1, :].detach()
                return hook_fn

            module.register_forward_hook(make_hook(name))

        if not self.ffn_module_names:
            print(f"警告: 未找到任何匹配 '{target_module_suffix}' 的模块。")
        else:
            print(f"已为 {len(self.ffn_module_names)} 个前馈网络模块注册钩子（仅跟踪最后一个token）")

    @torch.inference_mode()
    def get_activations(self, processed_inputs):
        """触发一次前向传播以获取最后一个token的激活"""
        self.activations = {}
        _ = self.model(**processed_inputs)
        return self.activations

    def filter_positive_activations(self, last_token_index):
        """
        筛选最后一个token上激活值 > 0 且 <= activation_cap 的神经元，
        并按top-k截断。
        """
        positive_activations = []

        for module_name, act_values in self.activations.items():  # act_values: [batch, hidden]
            k = min(self.top_k_per_layer, act_values.shape[-1])
            values, indices = torch.topk(act_values, k, dim=-1)
            mask = (values > 0) & (values <= self.activation_cap)
            pos_values = values[mask].float()  # 转为 float32 以兼容 numpy
            pos_indices = indices[mask]
            batch_indices = (
                torch.arange(act_values.shape[0], device=act_values.device)
                .unsqueeze(1)
                .expand(-1, k)[mask]
            )

            positive_activations.extend([
                {
                    'module': module_name,
                    'batch': int(b),
                    'seq': int(last_token_index),
                    'neuron': int(n),
                    'value': float(v)
                }
                for b, n, v in zip(
                    batch_indices.cpu().numpy(),
                    pos_indices.cpu().numpy(),
                    pos_values.cpu().numpy()
                )
            ])

        print(f"共筛选出 {len(positive_activations)} 个最后token激活值在 (0, {self.activation_cap}] 的神经元")
        return positive_activations

    def split_into_ranges(self, positive_activations, num_ranges=5):
        """将正激活值按大小分区"""
        if not positive_activations:
            return [], []

        values = np.array([x['value'] for x in positive_activations])
        percentiles = np.linspace(0, 100, num_ranges + 1)
        boundaries = np.percentile(values, percentiles)
        boundaries = np.unique(boundaries)

        ranges = []
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            range_neurons = [
                neuron for neuron in positive_activations
                if lower <= neuron['value'] <= upper
            ]
            ranges.append(
                {'range': (float(lower), float(upper)), 'neurons': range_neurons, 'count': len(range_neurons)}
            )

        actual_num_ranges = len(boundaries) - 1
        if actual_num_ranges < num_ranges:
            print(f"由于激活值分布，范围数量调整为: {actual_num_ranges}")

        print(f"将神经元分为 {actual_num_ranges} 个范围")
        return ranges, boundaries

    def _group_indices_by_module(self, neurons_to_perturb):
        grouped = defaultdict(list)
        for neuron in neurons_to_perturb:
            grouped[neuron['module']].append((neuron['batch'], neuron['neuron']))
        return grouped

    def perturb_and_generate_text(self, processed_inputs, neurons_to_perturb,
                                  perturbation_strength=0.5, max_new_tokens=30):
        """扰动最后 token 的指定神经元：直接将对应激活置 0"""
        perturbation_hooks = []

        if neurons_to_perturb:
            grouped = self._group_indices_by_module(neurons_to_perturb)

            for module_name, positions in grouped.items():
                module = self._get_module_by_name(self.model, module_name)
                if module is None:
                    continue

                # shape: [num_selected, 2] -> (batch_idx, neuron_idx)
                pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)

                def make_hook(current_module_name, pos_tensor):
                    def hook_fn(module, inputs, output):
                        tensor_to_perturb = output[0] if isinstance(output, tuple) else output
                        # tensor_to_perturb: [batch, seq, hidden]
                        last_token = tensor_to_perturb[:, -1, :]  # view
                        # 将指定 batch/神经元位置的激活直接置 0
                        last_token[pos_tensor[:, 0], pos_tensor[:, 1]] = 0.0
                        return output

                    return hook_fn

                hook = module.register_forward_hook(make_hook(module_name, pos_tensor))
                perturbation_hooks.append(hook)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                processed_inputs['input_ids'],
                attention_mask=processed_inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )

        for hook in perturbation_hooks:
            hook.remove()

        input_len = processed_inputs['input_ids'].shape[1]
        generated_part_ids = generated_ids[:, input_len:]
        generated_text = self.tokenizer.batch_decode(generated_part_ids, skip_special_tokens=True)[0]

        return generated_text.strip()

    def _get_module_by_name(self, model, name):
        for n, m in model.named_modules():
            if n == name:
                return m
        return None

    def evaluate_rouge_l(self, generated_text, reference_text):
        """计算ROUGE-L"""
        if not generated_text or not reference_text:
            return 0.0
        scores = self.scorer.score(reference_text, generated_text)
        return scores['rougeL'].fmeasure

    def run_experiment(self, prompt_text, reference_text,
                       num_ranges=5, perturbation_strength=0.5,
                       max_new_tokens=30):
        """运行完整实验：仅研究最后一个token的神经元，并剔除激活值 > activation_cap"""
        messages = [{"role": "assistant",
                     "content": "You are a medical expert. Given an input and an instruction, your objective is to respond with the correct and concise answer based on the provided context. \n\n### Guidelines for your response:\n\n1. **Ensure your responses are concise, clear, and focused on the provided instruction.** \n\n2. **Follow the logical order of questions.** Do not skip or merge responses.\n\n3. **Avoid adding extra commentary or irrelevant details. DO NOT repeat or summarize the question.** \n\n"},
                    {"role": "user", "content": prompt_text}]
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"使用的完整Prompt: {full_prompt}")

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        last_token_index = inputs['input_ids'].shape[1] - 1
        print(f"参考答案: '{reference_text}'")

        print("\n步骤1: 获取原始ROUGE-L分数 (无扰动)...")
        original_generated_text = self.perturb_and_generate_text(
            inputs, [], perturbation_strength=0, max_new_tokens=max_new_tokens
        )
        original_rouge_score = self.evaluate_rouge_l(original_generated_text, reference_text)
        print(f"原始生成文本: '{original_generated_text}'")
        print(f"原始ROUGE-L F1分数: {original_rouge_score:.4f}")

        print("\n步骤2: 获取最后token的神经元激活值...")
        self.get_activations(inputs)

        print("\n步骤3: 筛选最后token的正激活神经元 (去除 > {self.activation_cap} 的激活)...")
        positive_activations = self.filter_positive_activations(last_token_index)
        if not positive_activations:
            print("没有找到满足条件的神经元，实验结束")
            return None

        print("\n步骤4: 将神经元分为不同范围...")
        ranges, boundaries = self.split_into_ranges(positive_activations, num_ranges)

        print("\n步骤5: 对每个范围的神经元进行扰动实验...")
        results = []
        for i, range_data in enumerate(tqdm(ranges, desc="扰动实验")):
            if not range_data['neurons']:
                results.append({
                    'range_index': i, 'range': range_data['range'], 'neuron_count': 0,
                    'perturbed_text': original_generated_text, 'rouge_score': original_rouge_score,
                    'rouge_decrease': 0.0
                })
                continue

            perturbed_text = self.perturb_and_generate_text(
                inputs,
                range_data['neurons'],
                perturbation_strength,
                max_new_tokens=max_new_tokens
            )

            rouge_score = self.evaluate_rouge_l(perturbed_text, reference_text)
            rouge_decrease = original_rouge_score - rouge_score

            results.append({
                'range_index': i,
                'range': range_data['range'],
                'neuron_count': range_data['count'],
                'perturbed_text': perturbed_text,
                'rouge_score': rouge_score,
                'rouge_decrease': rouge_decrease
            })

        for res in results:
            print(f"范围 {res['range_index'] + 1}: 神经元数量={res['neuron_count']}, "
                  f"生成文本='{res['perturbed_text']}', ROUGE-L={res['rouge_score']:.4f}, "
                  f"分数下降={res['rouge_decrease']:.4f}")

        max_impact_range = max(results, key=lambda x: x['rouge_decrease']) if results else None

        experiment_results = {
            'original_rouge_score': original_rouge_score,
            'original_generated_text': original_generated_text,
            'reference_text': reference_text,
            'ranges': results,
            'max_impact_range': max_impact_range,
            'boundaries': boundaries.tolist(),
            'activation_cap': self.activation_cap
        }

        return experiment_results

    def plot_results(self, experiment_results, save_path=None):
        """可视化实验结果"""
        if not experiment_results or not experiment_results['ranges']:
            print("没有可可视化的结果")
            return

        ranges = experiment_results['ranges']
        original_score = experiment_results['original_rouge_score']

        range_labels = [f"R{r['range_index'] + 1}\n({r['range'][0]:.2f}-{r['range'][1]:.2f})" for r in ranges]
        scores = [r['rouge_score'] for r in ranges]
        score_decreases = [r['rouge_decrease'] for r in ranges]

        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            print("警告: 未找到 'SimHei' 字体，中文可能无法正常显示。")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        ax1.bar(range_labels, scores, label='ROUGE-L Score After Perturbation', color='skyblue')
        ax1.axhline(y=original_score, color='r', linestyle='--', label=f'Original ROUGE-L Score ({original_score:.2f})')
        ax1.set_title('ROUGE-L Scores of Neurons with Different Activation Ranges After Perturbation (Higher is Better)', fontsize=14)
        ax1.set_ylabel('ROUGE-L F1 Score', fontsize=12)
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45, labelsize=10)

        ax2.bar(range_labels, score_decreases, color='salmon')
        ax2.set_title('Degradation of ROUGE-L Scores Caused by Perturbations to Neurons with Different Activation Ranges (Higher Value Indicates Greater Impact)', fontsize=14)
        ax2.set_ylabel('ROUGE-L Score Degradation', fontsize=12)
        ax2.tick_params(axis='x', rotation=45, labelsize=10)

        plt.tight_layout(pad=3.0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n结果图表已保存至 {save_path}")
        else:
            plt.show()


# 示例使用
if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    PROMPT_TEXT = (
        "Identify the symptoms of a specific disease and suggest potential treatment options.:Disease: Parkinson's Disease"
    )
    REFERENCE_TEXT = (
        "Symptoms of Parkinson's Disease may include tremors, stiffness, difficulty with balance and coordination, and cognitive changes. Treatment may involve medications such as levodopa, carbidopa, or dopamine agonists to improve motor symptoms, as well as physical therapy and lifestyle changes to improve overall quality of life."
    )
    NUM_RANGES = 5
    PERTURBATION_STRENGTH = 1.0
    SAVE_RESULTS = True
    SEED = 42
    NUM_LAYERS_TO_TRACK = 32
    TOP_K_PER_LAYER = 1024
    MAX_NEW_TOKENS = 60
    ACTIVATION_CAP = 100  # 新增：最大允许激活值

    analyzer = NeuronAnalyzer(
        model_name=MODEL_NAME,
        seed=SEED,
        num_layers_to_track=NUM_LAYERS_TO_TRACK,
        top_k_per_layer=TOP_K_PER_LAYER,
        activation_cap=ACTIVATION_CAP
    )

    results = analyzer.run_experiment(
        prompt_text=PROMPT_TEXT,
        reference_text=REFERENCE_TEXT,
        num_ranges=NUM_RANGES,
        perturbation_strength=PERTURBATION_STRENGTH,
        max_new_tokens=MAX_NEW_TOKENS
    )

    if results:
        print("\n===== 实验结果总结 =====")
        print(f"参考答案: '{results['reference_text']}'")
        print(f"原始生成文本: '{results['original_generated_text']}'")
        print(f"原始ROUGE-L分数: {results['original_rouge_score']:.4f}")

        if results['max_impact_range']:
            max_range = results['max_impact_range']
            print(f"\n影响最大的范围: 范围 {max_range['range_index'] + 1}")
            print(f"  激活值区间: {max_range['range'][0]:.4f} - {max_range['range'][1]:.4f}")
            print(f"  神经元数量: {max_range['neuron_count']}")
            print(f"  扰动后生成: '{max_range['perturbed_text']}'")
            print(f"  ROUGE-L分数下降量: {max_range['rouge_decrease']:.4f} "
                  f"(从 {results['original_rouge_score']:.4f} 降至 {max_range['rouge_score']:.4f})")

        analyzer.plot_results(results,
                              save_path="qwen2_last_token_neuron_perturbation_rouge_results.png" if SAVE_RESULTS else None)

        if SAVE_RESULTS:
            output_filename = "qwen2_last_token_neuron_experiment_rouge_results.json"
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n实验结果已保存至 {output_filename}")