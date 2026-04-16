# InsightHealth AI

**Explainable Machine Learning for Health Risk Assessment**

---
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange?style=for-the-badge">
 <img src="https://img.shields.io/badge/ML-LightGBM-green?style=for-the-badge&logo=lightgbm">
  <img src="https://img.shields.io/badge/Explainability-SHAP-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Frontend-Streamlit-ff4b4b?style=for-the-badge">
</p>

---
## 🌐 Live Demo

<p align="center">
  <a href="https://ininsighthealth-ai.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Launch%20App-InsightHealth%20AI-blue?style=for-the-badge&logo=streamlit" />
  </a>
</p>

---
## Overview

InsightHealth AI is a clinical screening application designed to analyze diabetes risk indicators based on population data. Most predictive models operate as "Black Boxes," providing a risk score without justifying the underlying logic. This project addresses the **Black Box Problem** by integrating **SHAP (SHapley Additive exPlanations)** to provide feature-level transparency for every prediction.

---

## The Vision

* **Interpretability (XAI):** Utilizing a game-theoretic approach to explain model outputs. The system identifies exactly which biological or lifestyle factors—such as BMI, Blood Pressure, or Age—are driving a user's risk profile.
* **System Resilience:** Implemented a cascading LLM fallback architecture (Gemini → Gemma) to ensure consistent availability of health insights, even during API quota exhaustion.
* **Contextual Guidance:** The application employs zone-aware logic, ensuring the AI’s communication style is clinically appropriate—supportive for low-risk results and direct for priority risk zones.

---

## Problem Statement

Predictive modeling in digital health faces specific challenges regarding trust and reliability:

* **The Interpretability Gap:** High-accuracy results often lack transparency, making it difficult for users to understand which habits to prioritize for improvement.
* **API Dependency:** Reliance on a single cloud-based LLM can lead to system failure. This project implements a local/alternative fallback to maintain uptime.
* **Signal vs. Noise:** Statistical noise in low-impact features can lead to confusing explanations. A robust thresholding mechanism was developed to filter these out.

---

## Technical Approach

### 1. Data & Modeling

* **Dataset:** Trained on the CDC’s Behavioral Risk Factor Surveillance System (BRFSS) dataset.
* **Model:** LightGBM Classifier chosen for its efficiency with tabular data and superior handling of categorical health indicators.
* **Preprocessing:** Includes automated BMI calculation and validation against realistic clinical ranges.

### 2. Explainable AI (XAI)

* Integrated **SHAP TreeExplainer** to calculate local feature contributions.
* Implemented a signal-to-noise threshold (0.02) to ensure the user is only presented with statistically significant health drivers.

### 3. LLM Pipeline Engineering

* **Primary Inference:** Gemini 1.5 Flash for nuanced health coaching and reasoning.
* **Fallback Logic:** Automated switch to Gemma models via a custom error-handling wrapper to ensure zero downtime during API limit hits.

---

## Features

* **Risk Categorization:** Real-time classification into Low, Moderate, Elevated, or Priority zones.
* **Feature Impact Analysis:** Breakdown of the top contributing factors impacting the user's specific health profile.
* **Actionable Next Steps:** Contextual guidance and preventative steps mapped to specific risk levels.
* **Resilient AI Insights:** A redundant backend that ensures health explanations are always generated.

---

## Tech Stack

* **Machine Learning:** Scikit-learn, LightGBM, SHAP
* **LLMs:** Google Gemini API, Gemma (Fallback)
* **Frontend:** Streamlit
* **Environment:** Python 3.10+, Dotenv, Pandas, NumPy

---

## Project Structure

```bash
InsightHealth_AI/
├── app.py              # Main Streamlit interface
├── .gitignore          # Version control exclusions
├── .env                # API configuration (Local only)
├── src/
│   ├── ml_core.py      # Prediction and SHAP logic
│   └── insights.py     # LLM integration & fallback architecture
├── artifacts/
│   ├── best_model.joblib
│   └── features.joblib
└── notebooks/
    └── diabetes.ipynb  # EDA and model training
```

---

## Installation & Setup

1. **Clone Repository**

```bash
git clone https://github.com/PranayBothra/Insighthealth-ai.git
cd Insighthealth-ai
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure API Key**

Create a `.env` file in the root directory:

```text
GEMINI_API_KEY=your_actual_key_here
```

4. **Launch Application**

```bash
streamlit run app.py
```

---

## Disclaimer

*This application is a behavioral screening tool intended for educational purposes only. It does not provide medical diagnoses or treatment plans. Users should consult a qualified healthcare professional for any medical concerns.*
