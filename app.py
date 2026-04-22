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

def scan_clinical_report(text, expected_name):
    """
    Advanced Extraction Engine:
    Checks for the patient name first. Uses broader regex to catch age, 
    gender, HbA1c, diabetes history, and symptoms.
    """
    # 1. Verification Security Check
    if expected_name and expected_name.lower() not in text.lower():
        # This error is sent back to the frontend and displayed as an alert
        raise ValueError(f"Security Alert: The name '{expected_name}' was not found in the uploaded report. Extraction aborted.")

    extracted = {}

    # --- Demographics ---
    # Catches formats like "Age: 45", "Age/Gender: 45 Y", "45 Y / M"
    age_match = re.search(r"(?:Age|Age/Gender)\s*[:-]?\s*(\d+)", text, re.IGNORECASE)
    if age_match:
        extracted['age'] = age_match.group(1)

    if re.search(r"(?:Sex|Gender|Age/Gender).*?\b(Male|M)\b", text, re.IGNORECASE):
        extracted['gender'] = "Male"
    elif re.search(r"(?:Sex|Gender|Age/Gender).*?\b(Female|F)\b", text, re.IGNORECASE):
        extracted['gender'] = "Female"

    # --- Laboratory Results ---
    # Catches "HbA1c", "Glycosylated Hemoglobin", etc.
    hba1c_match = re.search(r"(?:HbA1c|Glycosylated Hemoglobin|Glycated Haemoglobin).*?(\d+\.\d+)", text, re.IGNORECASE | re.DOTALL)
    if hba1c_match:
        extracted['hba1c'] = hba1c_match.group(1)

    # --- Clinical History ---
    duration_match = re.search(r"(?:history of|duration of|known case of).*?(\d+)\s*(?:years?|yrs?)", text, re.IGNORECASE)
    if duration_match:
        extracted['diabetes_duration'] = duration_match.group(1)

    if re.search(r"\b(?:Type 1|Type I|T1DM)\b", text, re.IGNORECASE):
        extracted['diabetes_type'] = "Type 1"
    elif re.search(r"\b(?:Type 2|Type II|T2DM)\b", text, re.IGNORECASE):
        extracted['diabetes_type'] = "Type 2"

    # --- Symptoms ---
    if re.search(r"\b(?:Tingling|Numbness|Paresthesia)\b", text, re.IGNORECASE):
        extracted['tingling'] = "Yes"
    if re.search(r"\b(?:Burning|Burning Pain)\b", text, re.IGNORECASE):
        extracted['burning_pain'] = "Yes"
    if re.search(r"\b(?:Ulcer|Foot Ulcer|Wound)\b", text, re.IGNORECASE):
        extracted['foot_ulcer'] = "Yes"

    return extracted


# --- Frontend Routes ---
@app.route('/')
@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/symptoms.html')
def symptoms_page(): 
    return render_template('symptoms.html')

@app.route('/diagnosis.html')
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/treatment.html')
def treatment():
    return render_template('treatment.html')

@app.route('/prevention.html')
def prevention():
    return render_template('prevention.html')

@app.route('/prediction.html')
def prediction_page(): 
    return render_template('prediction.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')


# --- API Routes ---
@app.route("/extract", methods=["POST"])
def extract_report():
    if 'report' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    
    file = request.files['report']
    # Grab the patient name sent from the new Javascript
    patient_name = request.form.get('patient_name', '') 
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        with pdfplumber.open(filepath) as pdf:
            full_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        
        # Pass both text and name into the advanced scanner
        extracted_data = scan_clinical_report(full_text, patient_name)
        return jsonify({"success": True, "data": extracted_data})
    
    except ValueError as ve:
        # Catches the Name mismatch error specifically
        return jsonify({"success": False, "error": str(ve)})
    except Exception as e:
        return jsonify({"success": False, "error": "Failed to read PDF structure."})
    
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.json
        
        # Mapping exact HTML 'name' attributes to Model Feature Names
        mapping = {
            "age": "Age", 
            "gender": "Gender", 
            "bmi": "BMI",
            "smoking_status": "Smoking_Status", 
            "alcohol_status": "Alcohol_Status",
            "diabetes_type": "Diabetes_Type", 
            "diabetes_duration": "Diabetes_Duration_Years",
            "hba1c": "HbA1c_Level", 
            "tingling": "Symptom_Tingling_Numbness",
            "burning_pain": "Symptom_Burning_Pain", 
            "foot_ulcer": "History_Foot_Ulcer"
        }
        
        input_row = {}
        for json_key, feature_name in mapping.items():
            val = str(data.get(json_key, ""))
            
            # Numerical fields
            if feature_name in ["Age", "BMI", "Diabetes_Duration_Years", "HbA1c_Level"]:
                input_row[feature_name] = float(val) if (val and val.strip()) else 0.0
            # Categorical fields
            else:
                le = encoders[feature_name]
                valid_val = val if val in le.classes_ else le.classes_[0]
                input_row[feature_name] = le.transform([valid_val])[0]

        df_input = pd.DataFrame([input_row])[feature_cols]
        pred_index = model.predict(df_input)[0]
        risk_label = encoders["target"].inverse_transform([pred_index])[0]
        
        return jsonify({"success": True, "prediction": risk_label})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/submit_contact", methods=["POST"])
def submit_contact():
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')
        
        # In a real app, you would save this to a database or send an email.
        # For your project, printing it neatly to the terminal is perfect!
        print("\n" + "="*40)
        print("📩 NEW CONTACT FORM SUBMISSION")
        print("="*40)
        print(f"Name:    {name}")
        print(f"Email:   {email}")
        print(f"Subject: {subject}")
        print(f"Message: {message}")
        print("="*40 + "\n")
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)