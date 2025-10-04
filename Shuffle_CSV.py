import os
import pandas as pd
import numpy as np

# Input CSV and output folder
input_csv = "Processed_Data_2018/Merged.csv"
output_csv = "Processed_Data_2018/Merged_Shuffled_by_Label.csv"

# Parameters
chunk_size = 5_000_000  # adjust based on your RAM

# Step 1: Read CSV in chunks and group by label
print("Reading CSV and splitting by label...")
label_chunks = {}

for chunk in pd.read_csv(input_csv, chunksize=chunk_size, dtype=str, low_memory=False):
    for label, group in chunk.groupby('label'):
        if label not in label_chunks:
            label_chunks[label] = []
        label_chunks[label].append(group)

# Step 2: Shuffle rows within each label
for label in label_chunks:
    label_chunks[label] = [g.sample(frac=1, random_state=42) for g in label_chunks[label]]

# Step 3: Interleave rows from different labels
print("Interleaving rows from all labels...")
interleaved = []

# find the maximum number of chunks per label
max_chunks = max(len(chunks) for chunks in label_chunks.values())

for i in range(max_chunks):
    for label in label_chunks:
        if i < len(label_chunks[label]):
            interleaved.append(label_chunks[label][i])

# Concatenate all interleaved chunks
df_shuffled = pd.concat(interleaved).sample(frac=1, random_state=42)  # final shuffle

# Step 4: Save the shuffled CSV
df_shuffled.to_csv(output_csv, index=False)
print(f"Shuffled CSV saved to: {output_csv}")
