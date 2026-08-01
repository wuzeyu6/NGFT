#!/usr/bin/env python3
"""
Analyze the probability distributions for A, B, C, D in MMLU evaluation results.
"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict

print("=== Probability Distribution Analysis ===\n")

base_dir = "/Users/wu/Desktop/llama_factory/eval_result/mmlu"
bioinstruct_result_path = os.path.join(base_dir, "BioInstruct", "BioInstruct_Mistral-7B-Instruct_full")
base_result_path = os.path.join(base_dir, "base_model", "Mistral-7B-Instruct")

bio_subjects = [
    "college_biology", 
    "high_school_biology", 
    "professional_medicine", 
    "medical_genetics", 
    "virology",
    "anatomy",
    "nutrition",
    "clinical_knowledge"
]

choices = ["A", "B", "C", "D"]
probs_cols = [f"choice{c}_probs" for c in choices]

# Collect all probabilities
all_bioinstruct_probs = []
all_base_probs = []

for subject in bio_subjects:
    bioinstruct_file = os.path.join(bioinstruct_result_path, f"{subject}.csv")
    base_file = os.path.join(base_result_path, f"{subject}.csv")
    
    if os.path.exists(bioinstruct_file):
        bioinstruct_df = pd.read_csv(bioinstruct_file)
        base_df = pd.read_csv(base_file)
        
        all_bioinstruct_probs.append(bioinstruct_df[probs_cols].values)
        all_base_probs.append(base_df[probs_cols].values)

# Combine all probabilities
all_bioinstruct_probs = np.vstack(all_bioinstruct_probs)
all_base_probs = np.vstack(all_base_probs)

print(f"Total examples analyzed: {len(all_bioinstruct_probs)}")
print()

# 1. Average probabilities per choice
print("1. Average probabilities:")
bioinstruct_avg_probs = all_bioinstruct_probs.mean(axis=0)
base_avg_probs = all_base_probs.mean(axis=0)

for i, c in enumerate(choices):
    print(f"   {c}: BioInstruct={bioinstruct_avg_probs[i]:.3f}, Base={base_avg_probs[i]:.3f}")

# 2. Max probabilities - which option most frequently has max probability?
print("\n2. Max probability count (which option is chosen most often):")
bioinstruct_max_counts = (all_bioinstruct_probs.argmax(axis=1) == np.arange(4)[:, None]).sum(axis=1)
base_max_counts = (all_base_probs.argmax(axis=1) == np.arange(4)[:, None]).sum(axis=1)

for i, c in enumerate(choices):
    print(f"   {c}: BioInstruct={bioinstruct_max_counts[i]} ({bioinstruct_max_counts[i]/len(all_bioinstruct_probs)*100:.1f}%), Base={base_max_counts[i]} ({base_max_counts[i]/len(all_base_probs)*100:.1f}%)")

# 3. Let's look at some specific examples where B has very high probability
print("\n3. Examples where B has >90% probability (BioInstruct):")
high_b_indices = np.where(all_bioinstruct_probs[:, 1] > 0.90)[0]
print(f"   Found {len(high_b_indices)} examples where B > 90%")

if len(high_b_indices) > 0:
    # Let's get a few of these examples and see what the ground truth was
    # We need to collect ground truths
    gts = []
    for subject in bio_subjects:
        bioinstruct_file = os.path.join(bioinstruct_result_path, f"{subject}.csv")
        if os.path.exists(bioinstruct_file):
            df = pd.read_csv(bioinstruct_file)
            gts.extend(df.iloc[:, -2].values)
    
    print(f"\n   First 10 examples (index, B_prob, GT):")
    for idx in high_b_indices[:10]:
        print(f"   {idx}: B={all_bioinstruct_probs[idx,1]:.3f}, GT={gts[idx]}")

# 4. Analyze the distribution of B probabilities
print("\n4. B probability distribution in BioInstruct model:")
b_probs = all_bioinstruct_probs[:, 1]
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    print(f"   {p}th percentile: {np.percentile(b_probs, p):.3f}")

print(f"\n   Min: {b_probs.min():.3f}, Max: {b_probs.max():.3f}")
print(f"   Mean: {b_probs.mean():.3f}, Std: {b_probs.std():.3f}")
print(f"   Fraction of examples with B > 50%: {(b_probs > 0.5).mean():.1%}")
print(f"   Fraction of examples with B > 80%: {(b_probs > 0.8).mean():.1%}")
print(f"   Fraction of examples with B > 90%: {(b_probs > 0.9).mean():.1%}")

# 5. Compare with base model's B probabilities
print("\n5. B probability distribution in Base model:")
base_b_probs = all_base_probs[:, 1]
for p in percentiles:
    print(f"   {p}th percentile: {np.percentile(base_b_probs, p):.3f}")

print(f"\n   Min: {base_b_probs.min():.3f}, Max: {base_b_probs.max():.3f}")
print(f"   Mean: {base_b_probs.mean():.3f}, Std: {base_b_probs.std():.3f}")

print("\n=== Analysis Complete ===")
