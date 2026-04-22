// static/script.js
document.addEventListener('DOMContentLoaded', () => {

    // Contact Form Prevent Default
    const contactForm = document.getElementById('contact-form');
    if(contactForm) {
        contactForm.addEventListener('submit', (e) => {
            e.preventDefault();
            alert("Thank you for reaching out. We will get back to you shortly.");
            contactForm.reset();
        });
    }

    // Prediction & Extraction Logic
    const extractBtn = document.getElementById('extract-btn');
    const predictForm = document.getElementById('prediction-form');

    // 1. PDF Extraction
    if (extractBtn) {
        extractBtn.addEventListener('click', async () => {
            const fileInput = document.getElementById('pdfFile');
            const errorDiv = document.getElementById('extract-error');
            const successDiv = document.getElementById('extract-success');
            const spinner = document.getElementById('extract-spinner');

            errorDiv.classList.add('d-none');
            successDiv.classList.add('d-none');

            if (!fileInput.files.length) {
                errorDiv.textContent = "Please select a PDF file first.";
                errorDiv.classList.remove('d-none');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            extractBtn.disabled = true;
            spinner.classList.remove('d-none');

            try {
                // Call Flask /extract endpoint
                const response = await fetch('/extract', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error("Failed to extract data.");

                const data = await response.json();
                
                // Auto-fill form fields assuming JSON keys match
                if(data.age) document.getElementById('age').value = data.age;
                if(data.gender) document.getElementById('gender').value = data.gender;
                if(data.bmi) document.getElementById('bmi').value = data.bmi;
                if(data.diabetesType) document.getElementById('diabetesType').value = data.diabetesType;
                if(data.duration) document.getElementById('duration').value = data.duration;
                if(data.hba1c) document.getElementById('hba1c').value = data.hba1c;
                
                // Handle radio buttons
                if(data.smoking) {
                    if(data.smoking === 'Yes') document.getElementById('smokeYes').checked = true;
                    else if(data.smoking === 'No') document.getElementById('smokeNo').checked = true;
                }
                if(data.alcohol) {
                    if(data.alcohol === 'Yes') document.getElementById('alcYes').checked = true;
                    else if(data.alcohol === 'No') document.getElementById('alcNo').checked = true;
                }

                // Handle symptoms
                if(data.symptoms && Array.isArray(data.symptoms)) {
                    document.querySelectorAll('.symptom-cb').forEach(cb => {
                        if(data.symptoms.includes(cb.value)) cb.checked = true;
                    });
                }

                successDiv.classList.remove('d-none');
            } catch (err) {
                errorDiv.textContent = err.message || "An error occurred during extraction.";
                errorDiv.classList.remove('d-none');
            } finally {
                extractBtn.disabled = false;
                spinner.classList.add('d-none');
            }
        });
    }

    // 2. Prediction Submit
    if (predictForm) {
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = document.getElementById('predict-btn');
            const spinner = document.getElementById('predict-spinner');
            const errorDiv = document.getElementById('predict-error');
            const resultSection = document.getElementById('result-section');
            const resultCard = document.getElementById('result-card');
            
            errorDiv.classList.add('d-none');
            resultSection.classList.add('d-none');

            // Gather Symptoms
            const symptoms = [];
            document.querySelectorAll('.symptom-cb:checked').forEach(cb => {
                symptoms.push(cb.value);
            });

            // Gather Payload
            const payload = {
                age: document.getElementById('age').value,
                gender: document.getElementById('gender').value,
                bmi: document.getElementById('bmi').value,
                diabetesType: document.getElementById('diabetesType').value,
                duration: document.getElementById('duration').value,
                hba1c: document.getElementById('hba1c').value,
                smoking: document.querySelector('input[name="smoking"]:checked')?.value,
                alcohol: document.querySelector('input[name="alcohol"]:checked')?.value,
                symptoms: symptoms
            };

            submitBtn.disabled = true;
            spinner.classList.remove('d-none');

            try {
                // Call Flask /predict endpoint
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) throw new Error("Prediction API failed.");

                const data = await response.json();
                const risk = data.prediction || "Low"; // Expecting "Low", "Medium", "High"

                displayResults(risk);
                
                resultSection.classList.remove('d-none');
                
                // Scroll to result
                resultSection.scrollIntoView({ behavior: 'smooth' });

            } catch (err) {
                errorDiv.textContent = err.message || "Could not connect to the prediction server.";
                errorDiv.classList.remove('d-none');
            } finally {
                submitBtn.disabled = false;
                spinner.classList.add('d-none');
            }
        });
    }

    // Render logic for recommendations based on Risk Level
    function displayResults(riskLevel) {
        const badge = document.getElementById('risk-badge');
        const diet = document.getElementById('rec-diet');
        const lifestyle = document.getElementById('rec-lifestyle');
        const medical = document.getElementById('rec-medical');
        const card = document.getElementById('result-card');

        // Reset classes
        badge.className = 'result-badge d-inline-block px-5 py-3 rounded-pill text-white fw-bold h3 mb-4 shadow';
        card.className = 'card shadow border-0 border-top border-5';

        badge.textContent = `${riskLevel} Risk`;

        if (riskLevel.toLowerCase() === 'high') {
            badge.classList.add('badge-high');
            card.classList.add('border-danger');
            
            diet.innerHTML = "<strong>Strict Diabetic Diet:</strong> Severely restrict simple carbohydrates. High protein, high fiber. Avoid processed foods entirely.";
            lifestyle.innerHTML = "<strong>Cautious Activity:</strong> Avoid weight-bearing exercises if foot ulcers are present. Focus on upper body or swimming. Daily strict foot inspection required.";
            medical.innerHTML = "<strong>Immediate Consultation:</strong> Schedule an appointment with a neurologist and podiatrist immediately. Adjust medication for pain management and strict glycemic control.";
        } 
        else if (riskLevel.toLowerCase() === 'medium') {
            badge.classList.add('badge-medium');
            card.classList.add('border-warning');
            
            diet.innerHTML = "<strong>Controlled Diet:</strong> Monitor carbohydrate intake closely. Increase omega-3 fatty acids and antioxidants to support nerve health.";
            lifestyle.innerHTML = "<strong>Moderate Exercise:</strong> 30 mins of daily walking with proper orthotic footwear. Check feet every night before bed.";
            medical.innerHTML = "<strong>Follow-up Required:</strong> Discuss these results at your next endocrinologist visit. Consider comprehensive foot exam to establish a baseline.";
        } 
        else {
            badge.classList.add('badge-low');
            card.classList.add('border-success');
            
            diet.innerHTML = "<strong>Balanced Diet:</strong> Maintain a standard balanced diabetic diet. Stay hydrated to ensure good blood circulation.";
            lifestyle.innerHTML = "<strong>Active Lifestyle:</strong> Maintain regular exercise routines. Keep BMI within normal range. Basic foot hygiene is sufficient.";
            medical.innerHTML = "<strong>Routine Care:</strong> Continue with annual checkups and routine HbA1c monitoring. Keep blood sugars well within target range.";
        }
    }
});