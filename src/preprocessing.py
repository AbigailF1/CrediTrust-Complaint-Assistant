# src/preprocessing.py
# Preprocessing for CrediTrust Complaint Dataset

import pandas as pd
import re
import os

# -----------------------------
# 1. Load the full dataset
# -----------------------------
input_file = "data/raw/complaints.csv"
df = pd.read_csv(input_file, encoding="latin1")

print(f"Original dataset shape: {df.shape}")

# -----------------------------
# 2. Filter for target products
# -----------------------------
target_products = [
    "Credit card",
    "Personal loan",
    "Buy Now, Pay Later (BNPL)",
    "Savings account",
    "Money transfers"
]

df = df[df['Product'].isin(target_products)]
print(f"After filtering target products: {df.shape}")

# -----------------------------
# 3. Drop rows with empty narratives
# -----------------------------
df = df.dropna(subset=['Consumer complaint narrative'])
print(f"After dropping empty narratives: {df.shape}")

# -----------------------------
# 4. Clean text
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()  # lowercase
    text = re.sub(r'\s+', ' ', text)  # replace multiple whitespace with single space
    text = re.sub(r'[^a-z0-9 .,!?]', '', text)  # remove special chars (keep basic punctuation)
    text = text.strip()
    return text

df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_text)

# -----------------------------
# 5. Save cleaned dataset
# -----------------------------
os.makedirs("data/processed", exist_ok=True)
output_file = "data/processed/filtered_complaints.csv"
df.to_csv(output_file, index=False)

print(f"Preprocessed dataset saved to {output_file}")
