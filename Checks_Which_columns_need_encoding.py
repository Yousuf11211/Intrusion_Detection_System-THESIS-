import pandas as pd

csv_file = "Processed_Data_2017/Merged_Shuffled.csv"
chunk_size = 1_000_000

categorical_cols = set()

for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):
    cat_in_chunk = chunk.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.update(cat_in_chunk)

print("Categorical (string) columns:", list(categorical_cols))
