import pandas as pd

csv_file = "Downscale_Csv_2018/Cleaned.csv"
chunk_size = 1_000_000  # adjust based on memory

# Columns to check
columns_to_check = ['delta_start', 'handshake_duration', 'label']

# Store rows where handshake is invalid
invalid_rows = []

for chunk_idx, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size, usecols=columns_to_check, low_memory=False)):
    for col in ['delta_start', 'handshake_duration']:
        # Find rows where value is 'not a complete handshake'
        mask = chunk[col] == 'not a complete handshake'
        rows = chunk[mask]
        for idx, row in rows.iterrows():
            invalid_rows.append({
                'row_number': idx + chunk_idx * chunk_size,
                'column': col,
                'value': row[col],
                'label': row['label'],
                'both_invalid': row['delta_start'] == 'not a complete handshake' and row['handshake_duration'] == 'not a complete handshake'
            })

# Print the results
for entry in invalid_rows:
    print(f"Row {entry['row_number']}, Column: {entry['column']}, Value: {entry['value']}, Label: {entry['label']}, Both invalid: {entry['both_invalid']}")

print(f"\nTotal invalid rows found: {len(invalid_rows)}")
