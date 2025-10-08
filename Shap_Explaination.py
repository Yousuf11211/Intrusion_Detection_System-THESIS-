import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import shap

# -----------------------------
# FOLDER PATHS
# -----------------------------
model_folder = "Trained_Model_Complete_Data"
test_folder = "Test_Data"

model_path = os.path.join(model_folder, "random_forest_model.pkl")
label_mapping_path = os.path.join(model_folder, "label_mapping.txt")

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
rf = joblib.load(model_path)
print("Random Forest model loaded.")

# -----------------------------
# LOAD LABEL MAPPING
# -----------------------------
mapping = {}
with open(label_mapping_path, "r", encoding="utf-8") as f:
    lines = f.readlines()[2:]  # skip header
    for line in lines:
        if ":" in line:
            cls, num = line.strip().split(":")
            mapping[int(num.strip())] = cls.strip()
print("Label mapping loaded:", mapping)

# -----------------------------
# LOOP THROUGH ALL TEST CSVs
# -----------------------------
for file in os.listdir(test_folder):
    if not file.endswith(".csv"):
        continue

    test_path = os.path.join(test_folder, file)
    test_df = pd.read_csv(test_path)
    print(f"\nProcessing {file} ({test_df.shape[0]} rows)...")

    # -----------------------------
    # PREPARE FEATURES
    # -----------------------------
    X_test = test_df.drop(columns=['label'])

    # Encode categorical features if any
    for col in X_test.select_dtypes(include='object').columns:
        X_test[col] = LabelEncoder().fit_transform(X_test[col])

    # -----------------------------
    # PREDICTIONS
    # -----------------------------
    y_pred = rf.predict(X_test)
    y_pred_labels = [mapping[num] for num in y_pred]

    # -----------------------------
    # SHAP EXPLANATIONS
    # -----------------------------
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    feature_names = list(X_test.columns)

    attack_explanations = {}  # store one explanation per attack type

    for i, pred_num in enumerate(y_pred):
        attack_label = mapping[pred_num]

        # Only explain the first occurrence of each attack
        if attack_label not in attack_explanations:
            row_shap = shap_values[pred_num][i]
            contributions = sorted(
                zip(feature_names, row_shap),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            top_contributors = contributions[:3]
            explanation = f"{attack_label} predicted because: "
            explanation += ", ".join([f"{feat} was {'high' if val > 0 else 'low'}"
                                      for feat, val in top_contributors])
            attack_explanations[attack_label] = explanation

    # -----------------------------
    # PRINT EXPLANATIONS
    # -----------------------------
    print(f"\n--- Explanations for {file} ---")
    for attack, exp in attack_explanations.items():
        print(f"{attack:<20} → {exp}")
