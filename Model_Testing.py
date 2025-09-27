import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Paths
model_path = "Trained_Model_Complete_Data/random_forest_model.pkl"
label_mapping_path = "Trained_Model_Complete_Data/label_mapping.txt"
test_csv_path = "Test_Data/test_data.csv"

# Load trained model
rf = joblib.load(model_path)
print("Random Forest model loaded.")

# Load test data
test_df = pd.read_csv(test_csv_path)
print(f"Test data loaded: {test_df.shape[0]} samples.")

# Separate features and labels
X_test = test_df.drop(columns=['label'])
y_test = test_df['label']

# Encode categorical features (same as training)
for col in X_test.select_dtypes(include='object').columns:
    X_test[col] = LabelEncoder().fit_transform(X_test[col])

# Predict using the loaded model
y_pred = rf.predict(X_test)

# Load label mapping to decode numeric labels
mapping = {}
with open(label_mapping_path, "r", encoding="utf-8") as f:
    lines = f.readlines()[2:]  # skip header
    for line in lines:
        if ":" in line:
            cls, num = line.strip().split(":")
            mapping[int(num.strip())] = cls.strip()

# Convert predictions back to attack names
y_pred_labels = [mapping[num] for num in y_pred]

# Count how many times each attack was predicted
attack_counts = Counter(y_pred_labels)
print("Predicted attack counts:")
for attack, count in attack_counts.items():
    print(f"{attack:<20}: {count}")

# # Optional: Add predictions to the test DataFrame
# test_df['predicted_label'] = y_pred_labels
#
# # Save predictions
# pred_csv_path = "Test_Data/test_data_with_predictions.csv"
# test_df.to_csv(pred_csv_path, index=False)
# print(f"Predictions saved to {pred_csv_path}")
