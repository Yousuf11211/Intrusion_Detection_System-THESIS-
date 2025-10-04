import pandas as pd
import os

# Optional: Only needed for plots
try:
    import matplotlib.pyplot as plt

    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

FOLDER = "Training_2018"
FILENAME = "training_2_validated.csv"  # your cleaned data
FEATURE = "your_feature_column"  # change this to your feature of interest (e.g., 'packets_count')


def find_iqr_outliers(df, column):
    """Return boolean mask for IQR outliers in the given column"""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    return mask, lower, upper


def main():
    file_path = os.path.join(FOLDER, FILENAME)
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {FILENAME}")

    # Make sure feature exists and is numeric
    if FEATURE not in df.columns:
        print(f"Feature '{FEATURE}' not in columns!")
        return
    df[FEATURE] = pd.to_numeric(df[FEATURE], errors="coerce")

    # 1. Find outliers
    outlier_mask, lower, upper = find_iqr_outliers(df, FEATURE)
    print(f"IQR outlier thresholds for {FEATURE}:")
    print(f"  Lower: {lower:.2f}, Upper: {upper:.2f}")
    n_outliers = outlier_mask.sum()
    print(f"Found {n_outliers} outliers in '{FEATURE}'.")

    # 2. Check label distribution of outliers
    print("\nLabel counts among outliers:")
    outlier_labels = df.loc[outlier_mask, 'label'].value_counts()
    for label, count in outlier_labels.items():
        print(f"  {label}: {count}")

    # 3. Visualize per label (if matplotlib available)
    if HAS_PLOT:
        print("\nGenerating histogram by label...")
        plt.figure(figsize=(8, 4))
        for the_label in df['label'].unique():
            subset = df[df['label'] == the_label]
            subset[FEATURE].plot.hist(alpha=0.3, label=f"{the_label}", bins=30)
        plt.xlabel(FEATURE)
        plt.ylabel("Count")
        plt.title(f"Histogram of '{FEATURE}' by label")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("\n(Histogram skipped. To enable, run: pip install matplotlib)")

    # 4. Ask to tag outliers
    choice = input(f"\nDo you want to add an 'is_outlier_{FEATURE}' column and save new CSV? (y/n): ")
    if choice.lower() == 'y':
        out_col = f"is_outlier_{FEATURE}"
        df[out_col] = outlier_mask.astype(int)
        new_file = os.path.join(FOLDER, FILENAME.replace('.csv', f'_outlier_{FEATURE}.csv'))
        df.to_csv(new_file, index=False)
        print(f"New file with outlier tagging saved as: {os.path.basename(new_file)}")
    else:
        print("No changes were made or saved.")


if __name__ == "__main__":
    main()
