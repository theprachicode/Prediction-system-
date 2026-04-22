# ============================================================
# train_model.py  –  Clean + retrain version
# ============================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

print("Loading dataset...")
df = pd.read_csv("data/diabetes_data.csv")
df.drop(columns=["Patient_ID"], inplace=True)

# ── Standardise all categorical values ───────────────────────
# Fill NaN in Alcohol_Status → "None"  (NaN breaks LabelEncoder at inference)
df["Alcohol_Status"] = df["Alcohol_Status"].fillna("None")

# Normalise Yes/No columns
for col in ["Symptom_Tingling_Numbness", "Symptom_Burning_Pain", "History_Foot_Ulcer"]:
    df[col] = df[col].str.strip()
    df[col] = df[col].replace({"yes": "Yes", "y": "Yes", "Y": "Yes",
                                "no": "No",  "n": "No",  "N": "No"})

for col in ["Gender", "Smoking_Status", "Diabetes_Type"]:
    df[col] = df[col].str.strip()

# ── Encode ───────────────────────────────────────────────────
categorical_cols = [
    "Gender", "Smoking_Status", "Alcohol_Status", "Diabetes_Type",
    "Symptom_Tingling_Numbness", "Symptom_Burning_Pain", "History_Foot_Ulcer",
]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"  {col}: {list(le.classes_)}")

target_encoder = LabelEncoder()
df["Peripheral_Neuropathy_Risk"] = target_encoder.fit_transform(df["Peripheral_Neuropathy_Risk"])
encoders["target"] = target_encoder
print("Target:", list(target_encoder.classes_))

FEATURE_COLS = [
    "Age", "Gender", "BMI", "Smoking_Status", "Alcohol_Status", "Diabetes_Type",
    "Diabetes_Duration_Years", "HbA1c_Level",
    "Symptom_Tingling_Numbness", "Symptom_Burning_Pain", "History_Foot_Ulcer",
]

X, y = df[FEATURE_COLS], df["Peripheral_Neuropathy_Risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump({"model": model, "encoders": encoders, "feature_cols": FEATURE_COLS}, f)

print("✅ models/model.pkl saved!")
