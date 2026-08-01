#!/usr/bin/env python3
"""
Analyze the token encoding issue in MMLU evaluation.
The key part is how " A", " B", " C", " D" are encoded.
"""
import os
from transformers import AutoTokenizer

print("=== Token Encoding Analysis for MMLU Evaluation ===\n")

# Load Mistral tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test encoding of " A", " B", " C", " D"
choices = ["A", "B", "C", "D"]
print("\n1. Encoding analysis:")
for choice in choices:
    # MMLU code does: tokenizer.encode(" " + answer_choice, add_special_tokens=False)[-1]
    text = " " + choice
    encoded = tokenizer.encode(text, add_special_tokens=False)
    print(f"  '{text}' → {encoded} → last token: {encoded[-1] if encoded else 'N/A'}")
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"  Decoded back: '{decoded}'")
    
    # Also check encoding without space
    encoded_no_space = tokenizer.encode(choice, add_special_tokens=False)
    print(f"  '{choice}' (no space) → {encoded_no_space}")
    print()

# 2. Check token id correspondence
print("\n2. Token ID to Token mapping:")
for choice in choices:
    text = " " + choice
    encoded = tokenizer.encode(text, add_special_tokens=False)
    token_id = encoded[-1] if encoded else None
    token_text = tokenizer.decode([token_id]) if token_id else "N/A"
    print(f"  Token ID {token_id} → '{token_text}'")

# 3. Let's see what's in the tokenizer's vocab around those IDs
print("\n3. Checking vocab around token 330-340 (Mistral's B token is usually around there):")
for i in range(325, 345):
    try:
        token = tokenizer.decode([i])
        print(f"  ID {i}: '{token}'")
    except:
        pass

print("\n=== Analysis Complete ===")
