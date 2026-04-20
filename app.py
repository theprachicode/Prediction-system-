import os
import re
import pickle
import pandas as pd
import pdfplumber
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 1. Load the trained Machine Learning Model and Encoders
try:
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
        model = bundle["model"]
        encoders = bundle["encoders"]
        feature_cols = bundle["feature_cols"]
except FileNotFoundError:
    print("Error: model.pkl not found. Please ensure it is in the same directory.")

def scan_clinical_report(text):
    """
    Greedy Extraction Engine:
    Attempts to find all 11 model features within the PDF text.
    Tuned for clinical lab reports (like Amit Kulkarni's).
    """
    extracted = {}
    
    # --- 1. Demographics ---
    # Matches "Age/Gender : 50"
    age_match = re.search(r"Age/Gender\s*:\s*(\d+)", text, re.IGNORECASE)
    if age_match:
        extracted['age'] = age_match.group(1)

    # Detects Male/Female in the header context
    if re.search(r"Age/Gender\s*:.*?\bMale\b", text, re.IGNORECASE | re.DOTALL):
        extracted['gender'] = "Male"
    elif re.search(r"Age/Gender\s*:.*?\bFemale\b", text, re.IGNORECASE | re.DOTALL):
        extracted['gender'] = "Female"

    # --- 2. Laboratory Results ---
    # HbA1c (Matches: HbA1C- Glycated Haemoglobin ... 6.4)
    hba1c_match = re.search(r"HbA1C-.*?(\d+\.\d+)", text, re.IGNORECASE | re.DOTALL)
    if hba1c_match:
        extracted['hba1c'] = hba1c_match.group(1)

    # --- 3. Clinical History (If present in Remarks/Notes) ---
    # Diabetes Duration: Matches "History of diabetes for 10 years"
    duration_match = re.search(r"(?:History|Duration|Since).*?(\d+)\s*(?:Year|Yr|yr|year)", text, re.IGNORECASE)
    if duration_match:
        extracted['diabetes_duration'] = duration_match.group(1)

    # Diabetes Type: Matches "Type 1" or "Type 2"
    if re.search(r"Type\s*1|Type\s*I\b", text, re.IGNORECASE):
        extracted['diabetes_type'] = "Type 1"
    elif re.search(r"Type\s*2|Type\s*II\b", text, re.IGNORECASE):
        extracted['diabetes_type'] = "Type 2"

    # --- 4. Symptoms (If mentioned in clinical findings) ---
    if re.search(r"Tingling|Numbness|Paresthesia", text, re.IGNORECASE):
        extracted['tingling'] = "Yes"
    if re.search(r"Burning\s*Pain", text, re.IGNORECASE):
        extracted['burning_pain'] = "Yes"
    if re.search(r"Ulcer|Wound", text, re.IGNORECASE):
        extracted['foot_ulcer'] = "Yes"

    # Note: BMI and Lifestyle (Smoke/Alcohol) are almost never in blood reports.
    return extracted

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/extract", methods=["POST"])
def extract_report():
    if 'report' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    
    file = request.files['report']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        with pdfplumber.open(filepath) as pdf:
            # Join all pages as results can span multiple pages
            full_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        extracted_data = scan_clinical_report(full_text)
        return jsonify({"success": True, "data": extracted_data})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        mapping = {
            "age": "Age", "gender": "Gender", "bmi": "BMI",
            "smoking_status": "Smoking_Status", "alcohol_status": "Alcohol_Status",
            "diabetes_type": "Diabetes_Type", "diabetes_duration": "Diabetes_Duration_Years",
            "hba1c": "HbA1c_Level", "tingling": "Symptom_Tingling_Numbness",
            "burning_pain": "Symptom_Burning_Pain", "foot_ulcer": "History_Foot_Ulcer"
        }
        
        input_row = {}
        for key, feature in mapping.items():
            val = str(data.get(key, ""))
            # Numerical fields
            if feature in ["Age", "BMI", "Diabetes_Duration_Years", "HbA1c_Level"]:
                input_row[feature] = float(val) if (val and val.strip()) else 0.0
            # Categorical fields
            else:
                le = encoders[feature]
                valid_val = val if val in le.classes_ else le.classes_[0]
                input_row[feature] = le.transform([valid_val])[0]

        df_input = pd.DataFrame([input_row])[feature_cols]
        pred_index = model.predict(df_input)[0]
        risk_label = encoders["target"].inverse_transform([pred_index])[0]
        
        return jsonify({"success": True, "prediction": risk_label})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)