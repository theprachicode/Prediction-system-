// ============================================================
// static/script.js – Form handling & prediction display
// ============================================================

// ── Element references ──────────────────────────────────────
const form         = document.getElementById("predict-form");
const submitBtn    = document.getElementById("submit-btn");
const resetBtn     = document.getElementById("reset-btn");

const resultPanel  = document.getElementById("result-panel");
const loadingState = document.getElementById("loading-state");
const errorState   = document.getElementById("error-state");
const successState = document.getElementById("success-state");
const errorMsg     = document.getElementById("error-message");

const riskBadge    = document.getElementById("risk-badge");

// Probability bar elements
const bars = {
    Low:      { row: document.getElementById("bar-low"),      fill: document.querySelector(".low-fill") },
    Moderate: { row: document.getElementById("bar-moderate"), fill: document.querySelector(".mod-fill") },
    High:     { row: document.getElementById("bar-high"),     fill: document.querySelector(".high-fill") },
};

// ── Helpers ──────────────────────────────────────────────────

function showState(name) {
    if (!resultPanel) return;
    resultPanel.classList.remove("hidden");
    [loadingState, errorState, successState].forEach(el => el?.classList.add("hidden"));
    
    if (name === "loading") loadingState?.classList.remove("hidden");
    if (name === "error")   errorState?.classList.remove("hidden");
    if (name === "success") successState?.classList.remove("hidden");
}

function clearValidation() {
    form.querySelectorAll(".field-group").forEach(g => g.classList.remove("has-error"));
    form.querySelectorAll("input, select").forEach(el => el.classList.remove("invalid"));
}

/** Collect form values into a plain object */
function collectFormData() {
    const data = {};
    const fields = [
        "age", "gender", "bmi", "smoking_status", "alcohol_status", 
        "diabetes_type", "diabetes_duration", "hba1c", "tingling", 
        "burning_pain", "foot_ulcer"
    ];
    
    fields.forEach(id => {
        const el = document.getElementById(id);
        if (el) data[id] = el.value;
    });
    return data;
}

/** Animate probability bars */
function animateBars(probabilities) {
    // 🔍 Step 1: Find highest probability
    let maxLabel = null;
    let maxValue = -1;

    Object.entries(probabilities).forEach(([label, data]) => {
        const percentage = (typeof data === 'object') 
            ? data.percentage 
            : data;

        if (percentage > maxValue) {
            maxValue = percentage;
            maxLabel = label;
        }
    });

    // 🧹 Step 2: Hide all bars
    Object.values(bars).forEach(bar => {
        if (bar.row) bar.row.style.display = "none";
    });

    // ✅ Step 3: Show only highest probability
    if (bars[maxLabel]) {
        const fill = bars[maxLabel].fill;
        const pctLabel = bars[maxLabel].row.querySelector(".bar-pct");

        bars[maxLabel].row.style.display = "flex";

        if (fill) fill.style.width = maxValue + "%";
        if (pctLabel) pctLabel.textContent = Math.round(maxValue) + "%";
    }
}

/** Display prediction result */
function displayResult(data) {
    const label = data.prediction; // "Low" | "Moderate" | "High"

    riskBadge.textContent = label;
    riskBadge.className   = "risk-badge risk-" + label.toLowerCase();

    animateBars(data.probabilities);
    showState("success");

    resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Form submit ───────────────────────────────────────────────
if (form) {
    form.addEventListener("submit", async (e) => {
        e.preventDefault(); // CRITICAL: Prevents page refresh
        e.stopPropagation();

        clearValidation();
        const formData = collectFormData();

        // UI Feedback
        if (submitBtn) {
            submitBtn.disabled = true;
            const btnText = submitBtn.querySelector(".btn-text");
            if (btnText) btnText.textContent = "Assessing...";
        }
        showState("loading");

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(formData),
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                throw new Error(result.error || "Prediction failed.");
            }

            displayResult(result);

        } catch (err) {
            console.error(err);
            if (errorMsg) errorMsg.textContent = err.message;
            showState("error");
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                const btnText = submitBtn.querySelector(".btn-text");
                if (btnText) btnText.textContent = "Assess Risk";
            }
        }
    });
}

// ── Reset button ──────────────────────────────────────────────
if (resetBtn) {
    resetBtn.addEventListener("click", () => {
        resultPanel.classList.add("hidden");
        clearValidation();
        form.reset();
        window.scrollTo({ top: 0, behavior: "smooth" });
    });
}