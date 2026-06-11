import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings

# Suppress warnings to keep terminal clean
warnings.filterwarnings('ignore')

print("Loading dataset and applying preprocessing...")

# --- 1. YOUR EXACT DATA PREPROCESSING ---
df = pd.read_csv("Dibaties exact data - Copy.csv")

# Drop Patient_ID
df.drop(columns=["Patient_ID"], inplace=True, errors='ignore')

# Standardise categorical values
if "Alcohol_Status" in df.columns:
    df["Alcohol_Status"] = df["Alcohol_Status"].fillna("None")

# Normalise Yes/No columns
for col in ["Symptom_Tingling_Numbness", "Symptom_Burning_Pain", "History_Foot_Ulcer"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"yes": "Yes", "y": "Yes", "Y": "Yes", "no": "No",  "n": "No",  "N": "No", "nan": np.nan})

for col in ["Gender", "Smoking_Status", "Diabetes_Type"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan})

# Drop rows where the TARGET is missing (we can't train on a blank target)
df.dropna(subset=["Peripheral_Neuropathy_Risk"], inplace=True)

# Encode
categorical_cols = [
    "Gender", "Smoking_Status", "Alcohol_Status", "Diabetes_Type",
    "Symptom_Tingling_Numbness", "Symptom_Burning_Pain", "History_Foot_Ulcer",
]

for col in categorical_cols:
    if col in df.columns:
        # Fill missing categorical values with "Unknown" temporarily so LabelEncoder doesn't crash
        df[col] = df[col].fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

target_encoder = LabelEncoder()
df["Peripheral_Neuropathy_Risk"] = target_encoder.fit_transform(df["Peripheral_Neuropathy_Risk"])

FEATURE_COLS = [
    "Age", "Gender", "BMI", "Smoking_Status", "Alcohol_Status", "Diabetes_Type",
    "Diabetes_Duration_Years", "HbA1c_Level",
    "Symptom_Tingling_Numbness", "Symptom_Burning_Pain", "History_Foot_Ulcer",
]

X = df[FEATURE_COLS]
y = df["Peripheral_Neuropathy_Risk"]

# --- THE FIX: Fill any remaining blank numerical cells with the median ---
X = X.fillna(X.median())


# --- 2. TASK 1: MODEL COMPARISON ---
print("\n=========================================")
print("TASK 1: MODEL COMPARISON (80/20 Split)")
print("=========================================\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Support Vector Machine": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Random Forest (Your Model)": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1) 
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"{name}:")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  Precision: {prec*100:.1f}%")
    print(f"  Recall:    {rec*100:.1f}%")
    print(f"  F1-Score:  {f1*100:.1f}%\n")


# --- 3. TASK 2: TRAIN/TEST RATIO ANALYSIS ---
print("=========================================")
print("TASK 2: TRAIN/TEST RATIO ANALYSIS")
print("=========================================\n")

test_sizes = [0.10, 0.20, 0.25, 0.30, 0.40]
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

for test_size in test_sizes:
    X_train_ratio, X_test_ratio, y_train_ratio, y_test_ratio = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    rf_model.fit(X_train_ratio, y_train_ratio)
    y_pred_ratio = rf_model.predict(X_test_ratio)
    
    acc = accuracy_score(y_test_ratio, y_pred_ratio)
    
    train_pct = int((1 - test_size) * 100)
    test_pct = int(test_size * 100)
    print(f"Split {train_pct}% Train / {test_pct}% Test  -->  Accuracy: {acc*100:.2f}%")