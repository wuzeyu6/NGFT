<div align="center">

<img src="assets/logo.png" alt="NGFT Logo" width="200">

# Neuron-Guided Fine-Tuning (NGFT)

### Unlocking Efficient Alignment Mechanisms for Large Language Models

</div>

---

## Overview

**NGFT** (Neuron-Guided Fine-Tuning) is a holistic fine-tuning framework that leverages **neuron activation patterns** as a *universal proxy* to unify the fine-tuning lifecycle. Existing SFT paradigms suffer from parameter redundancy, inconsistent data quality, and catastrophic forgetting — issues typically addressed in isolation. NGFT bridges this gap by using neuron activation signals to jointly govern **data selection**, **parameter updates**, and **knowledge preservation**.

<div align="center">

| Key Result | Value |
|:---:|:---:|
| Domain Performance Gain (avg.) | **+4.66 pts** over FPFT |
| Forgetting Mitigation (avg.) | **+9.89 pts** over FPFT |
| Computational Cost | **~20.63%** of FPFT |

</div>

---

## Framework

NGFT consists of three synergistic mechanisms that cover the full fine-tuning lifecycle:

<div align="center">

<img src="assets/ngft_framework.png" alt="NGFT Framework" width="100%">

</div>

### (I) Task-Knowledge Neuron Selection

Identifies a sparse set of **Task-Knowledge Neurons** that are consistently and stably activated across the training data. Unlike fixed-ratio methods, NGFT adaptively calibrates optimal sparsity using the **Coefficient of Variation (CV)**:

- **Activation State Constraint**: Neurons with positive activation values respond effectively to input
- **Distribution Stability Constraint**: Low-CV neurons are stably activated and critical to performance
- **Gradient Masking**: Only key neuron parameters are updated; non-essential neurons are frozen

### (II) Activation-Based Data Selection

Selects high-quality instruction data based on **neuron activation intensity**. Samples near the median of the activation distribution strike the optimal balance between:

- **Sufficiency** — preserving task-relevant information
- **Minimality** — eliminating redundant noise

This median-centered strategy absorbs core samples from all sub-clusters, ensuring robustness to intra-class multimodal distributions.

### (III) Neuron Activation Alignment Loss

A novel loss function that aligns model activation patterns with **pre-computed activation anchors** derived from ground-truth responses:

- **Coarse-to-Fine Optimization**: Activated only after CE loss stabilizes (below threshold &gamma;)
- **Deep Knowledge Internalization**: Guides the model to learn neuron-level representations
- **Anti-Forgetting**: Preserves general knowledge by anchoring to pre-trained activation states

---

## Experimental Results

### Main Results

NGFT is evaluated on **Llama-3.1-8B**, **Qwen2.5-7B**, and **Mistral-7B** across three domain-specific datasets (GSM8K, BioInstruct, DialogSum) and three general benchmarks (MMLU, BBH, TyDiQA).


| Model | Method | GSM8K (Full) | DialogSum (Full) | BioInstruct (Full) |
|:---|:---|:---:|:---:|:---:|
| **Qwen2.5-7B** | FPFT | 84.91 / 58.09 | 37.78 / 57.88 | 40.21 / 58.10 |
| | Strong Baseline | 85.75 / 67.20 | 38.24 / 64.59 | 41.21 / 62.53 |
| | **NGFT (Ours)** | **87.64** / 67.10 | **39.41** / 64.21 | **42.27** / **65.34** |
| **Llama3.1-8B** | FPFT | 77.71 / 63.48 | 38.10 / 38.27 | 40.57 / 49.04 |
| | Strong Baseline | 77.64 / 68.25 | 38.58 / 48.30 | 40.84 / 55.05 |
| | **NGFT (Ours)** | **80.75** / **68.86** | **39.79** / **54.43** | **42.19** / **56.80** |
| **Mistral-7B** | FPFT | 57.24 / 47.20 | 36.42 / 14.02 | 38.14 / 12.80 |
| | Strong Baseline | 57.85 / 56.02 | 36.87 / 21.66 | 38.84 / 26.91 |
| | **NGFT (Ours)** | **62.55** / **58.07** | **38.06** / **27.71** | **39.60** / **30.77** |

</div>

> Format: **Test Score / CF Score**. Bold = Best. CF = average of MMLU, BBH, TyDiQA.

### Key Findings

1. **Superior Performance**: NGFT consistently outperforms FPFT and Strong Baseline across all models and datasets
2. **Low-Resource Excellence**: Using only 1%–10% data, NGFT achieves remarkable gains (e.g., +12.97 pts on GSM8K with Mistral-7B at 10% data)
3. **Efficiency**: NGFT achieves performance comparable to FPFT with only **20.63%** of the computational cost
4. **Anti-Forgetting**: NGFT significantly mitigates catastrophic forgetting, preserving general capabilities

---

## Quick Start

### 1. Environment Setup

```bash
conda create -n nsft python=3.10
conda activate nsft
cd llama_factory
pip install -e ".[torch,metrics]"
```

### 2. Run NGFT Pipeline

```bash
# GSM8K on Llama-3-8B
python pipeline_llama3_8/pipeline_GSM.py

# BioInstruct on Llama-3-8B
python pipeline_llama3_8/pipeline_Bio.py

# DialogSum on Llama-3-8B
python pipeline_llama3_8/pipeline_Dia.py
```

### 3. Data Selection (Standalone)

```bash
python data_selection.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --data_path data/gsm8k_dataset.json \
    --output_path data/selected_data.json \
    --sampling_ratio 0.05
```

### 4. Evaluation

```bash
# Full evaluation suite (MMLU + BBH + TyDiQA + domain-specific)
python all_eval.py --model_path model_result/GSM/GSM_Meta-Llama-3-8B-Instruct_full
```

---

## Project Structure

```
NGFT/
├── pipeline_llama3_1/         # Llama-3.1-8B pipelines
├── pipeline_llama3_3/         # Llama-3.3-8B pipelines
├── pipeline_llama3_8/         # Llama-3-8B pipelines
│   ├── pipeline_GSM.py         #   GSM8K training & eval
│   ├── pipeline_Bio.py         #   BioInstruct training & eval
│   ├── pipeline_Dia.py         #   DialogSum training & eval
│   └── pipeline_Bio_dw.py     #   BioInstruct with Data Whisperer
├── data/                       # Datasets
│   ├── hh_rlhf_en/             #   HH-RLHF
│   ├── ultra_chat/             #   UltraChat
│   └── gsm8k_dataset.json      #   GSM8K
├── eval/                        # Evaluation scripts
│   ├── codex_humaneval/         #   HumanEval
│   └── eval/                     #   Multi-benchmark evals
│       ├── gsm/                  #     GSM8K
│       ├── mmlu/                 #     MMLU
│       ├── bbh/                  #     BBH
│       └── tydiqa/               #     TyDiQA
├── evaluation/                  # Chinese benchmarks
│   ├── ceval/                    #   C-Eval
│   ├── cmmlu/                    #   CMMLU
│   └── mmlu/                     #   MMLU
├── data_selection.py            # Neuron-based data selection
├── neuron_compare.py            # Neuron analysis utilities
├── delete_checkpoint.py         # Checkpoint management
├── all_eval.py                  # Full evaluation orchestrator
├── examples/                    # LLaMA-Factory configs
├── docker/                      # Docker configurations
│   ├── docker-cuda/              #   CUDA support
│   ├── docker-npu/               #   NPU support
│   └── docker-rocm/              #   ROCm support
└── assets/                      # Images and resources
```

---


## Acknowledgments

This project is built upon [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We thank the open-source community for their valuable contributions.
