import requests
import os
from dotenv import load_dotenv
from streamlit import secrets

# Secure API Key Loading
try:
    API_KEY = secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    load_dotenv()
    API_KEY = os.getenv("GEMINI_API_KEY")

def generate_ai_insights(df_impact, risk_zone):
    # SEPARATE MODEL LISTS FOR TARGETED PROMPTING
    gemini_models = [
        "gemini-3-flash",
        "gemini-2.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite"
    ]
    
    gemma_models = [
        "gemma-4-31b",
        "gemma-4-26b",
        "gemma-3-27b",
        "gemma-3-12b"
    ]

    # --- 1. EXTRACT SHAP FACTORS ---
    top_pos = df_impact[df_impact['Impact'] > 0].sort_values('Impact', ascending=False).head(3)
    top_neg = df_impact[df_impact['Impact'] < 0].sort_values('Impact').head(3)

    if top_pos.empty and top_neg.empty:
        return "Keep up the great work! Your current indicators are neutral/excellent. Continue your healthy habits."

    # Map raw features to readable names
    feature_names = {
        'BMI': 'Body Mass Index (BMI)',
        'Age': 'Age Group',
        'GenHlth': 'Overall Health Rating',
        'HighBP': 'High Blood Pressure',
        'HighChol': 'High Cholesterol',
        'PhysActivity': 'Physical Activity Level',
        'Smoker': 'Smoking History',
        'DiffWalk': 'Difficulty Walking/Mobility'
    }
    
    pos_str = ", ".join([feature_names.get(f, f) for f in top_pos['Feature'].tolist()]) if not top_pos.empty else "None"
    neg_str = ", ".join([feature_names.get(f, f) for f in top_neg['Feature'].tolist()]) if not top_neg.empty else "None"

# 2. DYNAMIC ZONE RULES 
    # We define strict behavioral guidelines based on how severe the risk is.
    if risk_zone in ["Elevated Risk", "Priority Screening Recommended"]:
        gemini_tone_focus = "TONE: Serious, clinical, and direct. FOCUS: Strongly advising medical evaluation. Do not be overly cheerful or congratulate them."
        gemma_rules = """
        - Bullet 1: Acknowledge the elevated risk status directly and professionally.
        - Bullet 2: Highlight the specific "Areas to improve" that need clinical attention.
        - Bullet 3: Explicitly recommend scheduling a routine medical checkup to discuss these indicators.
        """
    else:
        gemini_tone_focus = "TONE: Positive, supportive, and preventative. FOCUS: Encouraging good habits and offering gentle lifestyle tweaks."
        gemma_rules = """
        - Bullet 1: Congratulate them on their "Current good habits".
        - Bullet 2: Give one specific, actionable lifestyle tip to improve their "Areas to improve".
        - Bullet 3: Encourage them to maintain their overall routine.
        """

    #  3. GEMINI PROMPT (Nuanced, Persona-Driven) 
    gemini_prompt = f"""
    You are an empathetic but highly clinical health assistant. A user has completed a health screening.
    
    Result Zone: '{risk_zone}'
    Factors elevating risk: {pos_str}
    Healthy factors protecting them: {neg_str}
    
    CRITICAL INSTRUCTIONS:
    {gemini_tone_focus}
    
    Task: Write a 3-4 sentence response. 
    1. Base your entire approach on the "CRITICAL INSTRUCTIONS" above.
    2. Address the elevating risk factors practically.
    3. If the risk is Low/Moderate, you may praise their healthy factors. If Elevated/Priority, skip the praise and focus on next steps.
    4. Do not diagnose (use phrases like "indicators suggest") and never use ML/SHAP jargon.
    """

    # 4. GEMMA PROMPT (Strict, Rule-Based Template) 
    gemma_prompt = f"""
    Write exactly 3 bullet points of health advice based on the following data. Do not write an introduction or conclusion.
    
    Status: {risk_zone}
    Areas to improve: {pos_str}
    Current good habits: {neg_str}
    
    Rules:
    {gemma_rules}
    - Do not use medical jargon, SHAP, or mention algorithms.
    """

    #  4. EXECUTION PIPELINE 
    
    def call_api(model_name, prompt_text):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
        response = requests.post(url, json=payload)
        return response.json()

    # Phase 1: Try Gemini Models
    for model in gemini_models:
        try:
            data = call_api(model, gemini_prompt)
            if "candidates" in data:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif "error" in data:
                err_msg = data['error']['message'].lower()
                if "quota" in err_msg or "limit" in err_msg or data['error'].get('code') == 429:
                    continue # Try next Gemini
        except Exception:
            continue

    # Phase 2: Try Gemma Models (If Gemini fails)
    for model in gemma_models:
        try:
            data = call_api(model, gemma_prompt)
            if "candidates" in data:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif "error" in data:
                err_msg = data['error']['message'].lower()
                if "quota" in err_msg or "limit" in err_msg or data['error'].get('code') == 429:
                    continue # Try next Gemma
        except Exception:
            continue

    return "⚠️ High traffic detected. All AI service quotas are currently exhausted. Please focus on standard healthy habits and consult your doctor for personalized advice."