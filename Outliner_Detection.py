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
FOLDER = "Training_2018"             # Folder containing the CSV file
FILENAME = "training_2_validated.csv"  # File name
OUT_FOLDER = os.path.join(FOLDER, "outlier_plots")  # Folder to save results


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

    # Load dataset
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from '{FILENAME}'")

    # Create output folder if it doesn't exist
    os.makedirs(OUT_FOLDER, exist_ok=True)

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found for outlier detection.")
        return

    print(f"\nFound {len(numeric_cols)} numeric columns to analyze.")

    # Process each numeric column
    for col in numeric_cols:
        print(f"\nProcessing column: {col}")

        # Drop rows with missing values for this column
        df_col = df.dropna(subset=[col]).copy()

        if df_col[col].nunique() <= 1:
            print(f"  Skipping column '{col}' (not enough unique values).")
            continue

        # Find outliers
        outlier_mask, lower, upper = find_iqr_outliers(df_col, col)
        n_outliers = outlier_mask.sum()

        print(f"  IQR thresholds -> Lower: {lower:.2f}, Upper: {upper:.2f}")
        print(f"  Found {n_outliers} outliers in '{col}'.")

        # Save CSV with outlier flag
        if n_outliers > 0:
            out_col = f"is_outlier_{col}"
            df_col[out_col] = outlier_mask.astype(int)
            out_csv = os.path.join(OUT_FOLDER, f"{col}_outliers.csv")
            df_col[[col, out_col]].to_csv(out_csv, index=False)
            print(f"  Saved outlier data to: {out_csv}")
        else:
            print(f"  No outliers found for '{col}', skipping CSV save.")

        # Generate and save plot (if matplotlib available)
        if HAS_PLOT:
            plt.figure(figsize=(10, 6))
            if df_col[col].nunique() > 50:
                df_col[col].hist(bins=50, color="steelblue", edgecolor="black")
                plt.title(f"Histogram of '{col}'")
            else:
                counts = df_col[col].value_counts().sort_index()
                plt.bar(counts.index, counts.values, color="steelblue")
                plt.title(f"Bar Chart of '{col}' Value Counts")

            plt.axvline(lower, color='red', linestyle='--', label='Lower IQR Bound')
            plt.axvline(upper, color='green', linestyle='--', label='Upper IQR Bound')
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()

            out_plot = os.path.join(OUT_FOLDER, f"{col}_plot.png")
            plt.savefig(out_plot)
            plt.close()
            print(f"  Saved plot to: {out_plot}")
        else:
            print("  Plot skipped (matplotlib not installed).")

    print("\nAll columns processed. Outlier reports and plots are saved in the folder:")
    print(OUT_FOLDER)


# ======================
# Run the script
# ======================
if __name__ == "__main__":
    main()
