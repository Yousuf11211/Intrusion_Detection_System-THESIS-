import pandas as pd

csv_file = "Downscale_Csv_2018/Cleaned.csv"
chunk_size = 1_500_000

# Dictionary to hold unique values for each categorical column
unique_values = {}

for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):
    cat_in_chunk = chunk.select_dtypes(include=['object']).columns.tolist()
    for col in cat_in_chunk:
        if col not in unique_values:
            unique_values[col] = set()
        unique_values[col].update(chunk[col].dropna().unique())

# Print results
print("Categorical (string) columns and unique counts:\n")
for col, vals in unique_values.items():
    print(f"- {col}: {len(vals)} unique values")
    if len(vals) < 20:  # only print small categories
        print(f"   → {list(vals)}")
