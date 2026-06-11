import os
import re
import pickle
import pandas as pd
import pdfplumber
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration for temporary PDF uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------------------------------------------------
# 1. Load the trained Machine Learning Model and Encoders
# ---------------------------------------------------------
try:
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
        model = bundle["model"]
        encoders = bundle["encoders"]
        feature_cols = bundle["feature_cols"]
except FileNotFoundError:
    print("Error: model.pkl not found. Please ensure it is in the same directory.")

# ---------------------------------------------------------
# 2. Advanced PDF Extraction Engine
# ---------------------------------------------------------
def scan_clinical_report(text, expected_name):
    """
    Advanced Extraction Engine with Auto-Inference and Month-to-Year Conversion.
    """
    # 1. Verification Security Check (Case Insensitive)
    if expected_name and expected_name.strip().lower() not in text.lower():
        raise ValueError("Please ensure the name entered exactly matches the name on the PDF document.")

    extracted = {}
    age_val = None

    # --- Demographics ---
    # Primary Check: Looks strictly for "Age" or "Age/Sex" and prevents jumping over random words (like "Page 1")
    age_match = re.search(r"\bAge\b[^\w\d]*(?:Sex|Gender|Yrs|Years)?[^\w\d]*(\d{1,3})\b", text, re.IGNORECASE)
    
    # Fallback Check: Sometimes lab reports just say "58 Yrs" without using the word "Age"
    if not age_match:
        age_match = re.search(r"\b(\d{1,3})\s*(?:Yrs|Years|Y/O)\b", text, re.IGNORECASE)

    if age_match:
        age_val = int(age_match.group(1))
        if 1 <= age_val <= 120: 
            extracted['age'] = str(age_val)

    if re.search(r"(?:Sex|Gender|Age\s*/\s*Sex|Age\s*/\s*Gender).*?\b(Male|M)\b", text, re.IGNORECASE):
        extracted['gender'] = "Male"
    elif re.search(r"(?:Sex|Gender|Age\s*/\s*Sex|Age\s*/\s*Gender).*?\b(Female|F)\b", text, re.IGNORECASE):
        extracted['gender'] = "Female"

    # --- Laboratory Results ---
    hba1c_match = re.search(r"(?:HbA1c|Glycosylated Hemoglobin|Glycated Haemoglobin).*?(\d+\.\d+)", text, re.IGNORECASE | re.DOTALL)
    if hba1c_match:
        extracted['hba1c'] = hba1c_match.group(1)

    # --- Clinical History (Handles Months & Years) ---
    duration_match = re.search(r"(?:history of|duration of|known case of).*?(\d+)\s*(months?|yrs?|years?)", text, re.IGNORECASE)
    if duration_match:
        val = float(duration_match.group(1))
        unit = duration_match.group(2).lower()
        if 'month' in unit:
            val = round(val / 12.0, 2) # Convert months to years (e.g., 6 months -> 0.5 years)
        extracted['diabetes_duration'] = str(val)
        extracted['duration_unit'] = "years" # Force dropdown to match decimal

    # --- Type Inference ---
    if re.search(r"\b(?:Type 1|Type I|T1DM)\b", text, re.IGNORECASE):
        extracted['diabetes_type'] = "Type 1"
    elif re.search(r"\b(?:Type 2|Type II|T2DM)\b", text, re.IGNORECASE):
        extracted['diabetes_type'] = "Type 2"
    else:
        # Intelligent fallback: Infer Type by Age if not explicitly stated
        if age_val is not None:
            extracted['diabetes_type'] = "Type 1" if age_val < 30 else "Type 2"

    # --- Symptoms ---
    if re.search(r"\b(?:Tingling|Numbness|Paresthesia)\b", text, re.IGNORECASE):
        extracted['tingling'] = "Yes"
    if re.search(r"\b(?:Burning|Burning Pain)\b", text, re.IGNORECASE):
        extracted['burning_pain'] = "Yes"
    if re.search(r"\b(?:Ulcer|Foot Ulcer|Wound)\b", text, re.IGNORECASE):
        extracted['foot_ulcer'] = "Yes"

    return extracted

# ---------------------------------------------------------
# 3. Frontend Page Routes
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 4. API Endpoints (Extraction, Prediction, Contact)
# ---------------------------------------------------------

@app.route("/extract", methods=["POST"])
def extract_report():
    if 'report' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    
    file = request.files['report']
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
        # Catches the specific Name Mismatch error
        return jsonify({"success": False, "error": str(ve)})
    except Exception as e:
        return jsonify({"success": False, "error": "Failed to read PDF structure."})
    
    finally:
        # Always delete the uploaded file after processing to maintain security/statelessness
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.json
        
        # --- NEW: Convert Months to Years if user selected "Months" ---
        duration_val = float(data.get("diabetes_duration", 0))
        duration_unit = data.get("duration_unit", "years")
        if duration_unit == "months":
            final_duration = round(duration_val / 12.0, 2)
        else:
            final_duration = duration_val

        # Mapping exact HTML 'name' attributes to Model Feature Names
        mapping = {
            "age": "Age", 
            "gender": "Gender", 
            "bmi": "BMI",
            "smoking_status": "Smoking_Status", 
            "alcohol_status": "Alcohol_Status",
            "diabetes_type": "Diabetes_Type", 
            "hba1c": "HbA1c_Level", 
            "tingling": "Symptom_Tingling_Numbness",
            "burning_pain": "Symptom_Burning_Pain", 
            "foot_ulcer": "History_Foot_Ulcer"
        }
        
        input_row = {}
        for json_key, feature_name in mapping.items():
            val = str(data.get(json_key, ""))
            
            # Numerical fields
            if feature_name in ["Age", "BMI", "HbA1c_Level"]:
                input_row[feature_name] = float(val) if (val and val.strip()) else 0.0
            # Categorical fields
            else:
                le = encoders[feature_name]
                # Fallback to the first class if the value is unexpected/empty
                valid_val = val if val in le.classes_ else le.classes_[0]
                input_row[feature_name] = le.transform([valid_val])[0]

        # Insert the correctly calculated duration
        input_row["Diabetes_Duration_Years"] = final_duration

        # Convert dictionary to DataFrame using the exact column order expected by the model
        df_input = pd.DataFrame([input_row])[feature_cols]
        
        # Run inference
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
        
        # Print neatly to the terminal for presentation/testing proof
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