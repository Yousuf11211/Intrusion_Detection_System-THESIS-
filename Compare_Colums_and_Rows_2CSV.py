import pandas as pd

# Paths to the two CSV files
csv1_path = "Processed_Data_2017_Dask/Merged_Shuffled_All.csv"
csv2_path = "Processed_Data_2017/Merged_Shuffled.csv"

# Read CSVs
df1 = pd.read_csv(csv1_path, dtype=str)
df2 = pd.read_csv(csv2_path, dtype=str)

# ----- Check columns -----
if list(df1.columns) == list(df2.columns):
    print("Columns match exactly.")
else:
    print("Columns do NOT match.")
    print("Columns in CSV1 but not in CSV2:", set(df1.columns) - set(df2.columns))
    print("Columns in CSV2 but not in CSV1:", set(df2.columns) - set(df1.columns))

# ----- Check rows ignoring order -----
# Convert each row to tuple and make sets
rows1 = set(tuple(row) for row in df1.values)
rows2 = set(tuple(row) for row in df2.values)

if rows1 == rows2:
    print("All rows match (order ignored).")
else:
    print("Rows do NOT match.")
    print("Rows in CSV1 but not in CSV2:", len(rows1 - rows2))
    print("Rows in CSV2 but not in CSV1:", len(rows2 - rows1))
