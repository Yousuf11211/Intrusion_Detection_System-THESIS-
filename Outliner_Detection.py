import pandas as pd
import os

# Try to import matplotlib for optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ======================
# Configuration
# ======================
FOLDER = "Training_2018"            # Folder containing the CSV file
FILENAME = "training_2_validated.csv"  # File name
FEATURE = "src_port"                # Feature (column) to analyze for outliers


# ======================
# Helper function
# ======================
def find_iqr_outliers(df, column):
    """Find outliers in a numeric column using the IQR (Interquartile Range) method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    return mask, lower, upper


# ======================
# Main function
# ======================
def main():
    # Build full file path
    file_path = os.path.join(FOLDER, FILENAME)

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Load the dataset
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from '{FILENAME}'")

    # Make sure the feature exists
    if FEATURE not in df.columns:
        print(f"Error: Feature '{FEATURE}' not found in the dataset columns.")
        print("Available columns are:")
        print(df.columns.tolist())
        return

    # Convert the feature to numeric (in case it contains strings)
    df[FEATURE] = pd.to_numeric(df[FEATURE], errors="coerce")

    # Drop rows where the feature is missing
    df = df.dropna(subset=[FEATURE])

    # --- Find outliers ---
    outlier_mask, lower, upper = find_iqr_outliers(df, FEATURE)
    print(f"\nIQR thresholds for '{FEATURE}':")
    print(f"   Lower bound: {lower:.2f}")
    print(f"   Upper bound: {upper:.2f}")

    n_outliers = outlier_mask.sum()
    print(f"Found {n_outliers} outliers in '{FEATURE}'.")

    # --- Check label distribution among outliers ---
    if 'label' in df.columns and n_outliers > 0:
        print("\nLabel counts among outliers:")
        outlier_labels = df.loc[outlier_mask, 'label'].value_counts()
        for label, count in outlier_labels.items():
            print(f"   {label}: {count}")
    elif 'label' not in df.columns:
        print("\nNo 'label' column found, skipping label distribution check.")
    else:
        print("\nNo outliers detected.")

    # --- Visualization (optional) ---
    if HAS_PLOT:
        print("\nGenerating plot...")

        plt.figure(figsize=(10, 6))
        if df[FEATURE].nunique() > 50:
            # Use a histogram for continuous features
            df[FEATURE].hist(bins=50, color="steelblue", edgecolor="black")
            plt.title(f"Histogram of '{FEATURE}'")
        else:
            # Use a bar chart for discrete values
            counts = df[FEATURE].value_counts().sort_index()
            plt.bar(counts.index, counts.values, color="steelblue")
            plt.title(f"Bar Chart of '{FEATURE}' Value Counts")

        plt.axvline(lower, color='red', linestyle='--', label='Lower IQR Bound')
        plt.axvline(upper, color='green', linestyle='--', label='Upper IQR Bound')
        plt.xlabel(FEATURE)
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("\n(Plot skipped. Install matplotlib to enable plotting: pip install matplotlib)")

    # --- Ask user if they want to save the results ---
    if n_outliers > 0:
        choice = input(f"\nDo you want to save a new CSV with outlier flags for '{FEATURE}'? (y/n): ")
        if choice.lower() == 'y':
            out_col = f"is_outlier_{FEATURE}"
            df[out_col] = outlier_mask.astype(int)
            new_file = os.path.join(FOLDER, FILENAME.replace('.csv', f'_outlier_{FEATURE}.csv'))
            df.to_csv(new_file, index=False)
            print(f"New file saved as: {os.path.basename(new_file)}")
        else:
            print("No file saved.")
    else:
        print("\nNo outliers to tag. Nothing saved.")


# ======================
# Run the script
# ======================
if __name__ == "__main__":
    main()
