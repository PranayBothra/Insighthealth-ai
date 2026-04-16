import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path
import streamlit as st

# Setup dynamic pathing to root directory
BASE_DIR = Path(__file__).resolve().parents[1]

@st.cache_resource
def load_explainer(model):
    return shap.TreeExplainer(model)
@st.cache_resource
def load_assets():
    # Make sure your artifacts are placed correctly relative to this script
    model = joblib.load(BASE_DIR / 'best_model.joblib')
    features = joblib.load(BASE_DIR / 'features.joblib')
    return model, features

def map_age(raw_age):
    if raw_age <= 24: return 1
    elif raw_age <= 29: return 2
    elif raw_age <= 34: return 3
    elif raw_age <= 39: return 4
    elif raw_age <= 44: return 5
    elif raw_age <= 49: return 6
    elif raw_age <= 54: return 7
    elif raw_age <= 59: return 8
    elif raw_age <= 64: return 9
    elif raw_age <= 69: return 10
    elif raw_age <= 74: return 11
    elif raw_age <= 79: return 12
    else: return 13

def preprocess_inputs(input_data, feature_cols):
    bmi_calc = round(input_data['weight'] / ((input_data['height'] / 100) ** 2), 1)
    
    gen_hlth_map = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}
    binary_map = {"No": 0, "Yes": 1}

    input_dict = {
        'BMI': bmi_calc,
        'Age': map_age(input_data['age']),
        'GenHlth': gen_hlth_map[input_data['gen_hlth']],
        'HighBP': binary_map[input_data['high_bp']],
        'HighChol': binary_map[input_data['high_chol']],
        'PhysActivity': binary_map[input_data['phys_act']],
        'Smoker': binary_map[input_data['smoker']],
        'DiffWalk': binary_map[input_data['diff_walk']]
    }

    return pd.DataFrame([input_dict], columns=feature_cols)

def explain_prediction(model, input_df):
    probability = model.predict_proba(input_df)[0][1]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)
    
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1][0] 
    else:
        shap_vals = shap_vals[0]

    df_impact = pd.DataFrame({
        'Feature': input_df.columns,
        'Impact': shap_vals
    }).sort_values('Impact', ascending=False)

    return probability, df_impact


# ZONE-AWARE SHAP TEXT GENERATOR (HTML FORMATTED)
def generate_text_explanation(df_impact, probability):
    # Map raw features to user-friendly names
    feature_names = {
        'BMI': 'Body Mass Index (BMI)',
        'Age': 'Age Group',
        'GenHlth': 'Overall Health Rating',
        'HighBP': 'Blood Pressure History',
        'HighChol': 'Cholesterol History',
        'PhysActivity': 'Physical Activity Level',
        'Smoker': 'Smoking History',
        'DiffWalk': 'Mobility / Walking Ability'
    }

    # 🔥 CHANGE 1: Define a threshold (0.02 is usually safe to filter noise)
    threshold = 0.02

    # 🔥 CHANGE 2: Filter by threshold before taking the head(3)
    top_pos = df_impact[df_impact['Impact'] > threshold].sort_values('Impact', ascending=False).head(3)
    top_neg = df_impact[df_impact['Impact'] < -threshold].sort_values('Impact').head(3)

    explanation = ""

    # Logic 1: LOW RISK (< 0.3)
    if probability < 0.3:
        if not top_neg.empty:
            explanation += "<b>🟢 What's keeping you in the safe zone:</b><br>"
            for _, row in top_neg.iterrows():
                explanation += f"&bull; Excellent baseline for <b>{feature_names.get(row['Feature'], row['Feature'])}</b><br>"
        if not top_pos.empty:
            explanation += "<br><i>Minor areas to monitor (currently not a concern):</i><br>"
            for _, row in top_pos.iterrows():
                explanation += f"&bull; {feature_names.get(row['Feature'], row['Feature'])}<br>"

    # Logic 2: MODERATE RISK (0.3 - 0.5)
    elif probability < 0.5:
        if not top_pos.empty:
            explanation += "<b>🟡 Early warning signs to improve:</b><br>"
            for _, row in top_pos.iterrows():
                explanation += f"&bull; <b>{feature_names.get(row['Feature'], row['Feature'])}</b> is slightly elevating your profile.<br>"
        if not top_neg.empty:
            explanation += "<br><b>🟢 Healthy habits protecting you:</b><br>"
            for _, row in top_neg.iterrows():
                explanation += f"&bull; {feature_names.get(row['Feature'], row['Feature'])}<br>"

    # Logic 3: ELEVATED RISK (0.5 - 0.7)
    elif probability < 0.7:
        if not top_pos.empty:
            explanation += "<b>🟠 Key factors elevating your risk:</b><br>"
            for _, row in top_pos.iterrows():
                explanation += f"&bull; <b>{feature_names.get(row['Feature'], row['Feature'])}</b> strongly contributes to this result.<br>"
        if not top_neg.empty:
            explanation += "<br><b>🟢 Mitigating factors (What you're doing right):</b><br>"
            for _, row in top_neg.iterrows():
                explanation += f"&bull; {feature_names.get(row['Feature'], row['Feature'])}<br>"

    # Logic 4: PRIORITY SCREENING (> 0.7)
    else:
        if not top_pos.empty:
            explanation += "<b>🔴 Primary clinical indicators detected:</b><br>"
            for _, row in top_pos.iterrows():
                explanation += f"&bull; <b>{feature_names.get(row['Feature'], row['Feature'])}</b> aligns closely with risk profiles.<br>"
        explanation += "<br><i>Please discuss these specific points with your doctor during your screening.</i>"

    if explanation == "":
        explanation = "Your indicators are completely neutral."

    return explanation