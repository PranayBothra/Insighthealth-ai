import streamlit as st
import time

# Import from our modular backend
from src.ml_core import load_assets, preprocess_inputs, explain_prediction, generate_text_explanation
from src.insights import generate_ai_insights
# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Health Indicator Screening",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. LOAD ASSETS ---
with st.spinner("Initializing system..."):
    model, feature_cols = load_assets()
    if model is None:
        st.error("Failed to load ML model or features. Check your 'best_model.joblib' file.")
        st.stop()

# --- 3. SESSION STATE MANAGEMENT ---
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'probability' not in st.session_state:
    st.session_state.probability = None
if 'df_impact' not in st.session_state:
    st.session_state.df_impact = None
if 'zone' not in st.session_state: 
    st.session_state.zone = None

# --- 4. APP HEADER ---
st.title("🩺 Health Indicator Screening")
st.markdown("""
**Disclaimer:** *This is a behavioral screening tool based on CDC data, not a medical diagnostic device. Please consult a healthcare professional for actual medical advice.*
""")
st.markdown("""
 *This tool identifies patterns, not diagnoses.*
""")
st.divider()

# --- 5. UI FORM ---
with st.form("health_form"):
    st.subheader("1. Basic Vitals")
    col1, col2, col3 = st.columns(3)
    with col1:
        age_input = st.slider("Age", 18, 100, 40)
    with col2:
        height_cm = st.number_input("Height (cm)", 100, 250, 170)
    with col3:
        weight_kg = st.number_input("Weight (kg)", 30, 200, 70)

    st.subheader("2. Clinical History")
    col4, col5 = st.columns(2)
    with col4:
        gen_hlth_input = st.selectbox("Overall, how would you rate your health?", ["Excellent", "Very Good", "Good", "Fair", "Poor"], index=2)
        high_bp_input = st.radio("Ever told you have High Blood Pressure?", ["No", "Yes"], horizontal=True)
    with col5:
        high_chol_input = st.radio("Ever told you have High Cholesterol?", ["No", "Yes"], horizontal=True)
        diff_walk_input = st.radio("Serious difficulty walking or climbing stairs?", ["No", "Yes"], horizontal=True)

    st.subheader("3. Lifestyle")
    col6, col7 = st.columns(2)
    with col6:
        phys_act_input = st.radio("Physical activity/exercise in past 30 days?", ["Yes", "No"], horizontal=True)
    with col7:
        smoker_input = st.radio("Smoked 100+ cigarettes in your life?", ["No", "Yes"], horizontal=True)

    submit_button = st.form_submit_button("Analyze Health Indicators", use_container_width=True)

# --- 6. EXECUTION LOGIC ---
if submit_button:
    if "ai_text" in st.session_state:
        del st.session_state.ai_text
    bmi_calc = round(weight_kg / ((height_cm / 100) ** 2), 1)
    
    if bmi_calc < 12 or bmi_calc > 60:
        st.error("⚠️ **Input Error:** The calculated BMI is outside realistic clinical ranges. Please verify your height and weight.")
        st.stop()
    with st.spinner("Analyzing your health indicators..."):
        time.sleep(1.5) # UX friction for clinical credibility
        
        raw_inputs = {
            'age': age_input, 
            'height': height_cm, 
            'weight': weight_kg,
            'gen_hlth': gen_hlth_input, 
            'high_bp': high_bp_input, 
            'high_chol': high_chol_input, 
            'phys_act': phys_act_input, 
            'smoker': smoker_input, 
            'diff_walk': diff_walk_input
        }
        
        input_df = preprocess_inputs(raw_inputs, feature_cols)
        prob, df_impact = explain_prediction(model, input_df)
        
        st.session_state.probability = prob
        st.session_state.df_impact = df_impact
        st.session_state.analyzed = True

# --- 7. RENDER RESULTS ---
if st.session_state.analyzed:
    prob = st.session_state.probability
    df_impact = st.session_state.df_impact
    
    st.divider()
    st.subheader("Assessment Results")

    # --- 4-ZONE UX ROUTING ---
    if prob < 0.3:
        st.session_state.zone = "Low Risk"
        st.success("### 🟢 Low Risk")
        st.write("**You are unlikely to have diabetes based on current indicators.**")
        st.write("Your health indicators are within a safe range. Continue maintaining a healthy lifestyle.")
        
    elif prob < 0.5:
        st.session_state.zone = "Moderate Risk"
        st.warning("### 🟡 Moderate Risk")
        st.write("**Early signs of potential risk are present.**")
        st.write("Some early indicators detected. Improving lifestyle habits can help reduce future risk.")
        
    elif prob < 0.7:
        st.session_state.zone = "Elevated Risk"
        st.error("### 🟠 Elevated Risk")
        st.write("**Your indicators suggest increased risk, but are not definitive.**")
        st.write("Elevated indicators detected. A routine medical checkup is recommended.")
        
    else:
        st.session_state.zone = "Priority Screening Recommended"
        st.markdown("""
        <div style="padding: 1.5rem; background-color: #ffcccc; color: #990000; border-radius: 0.5rem; margin-bottom: 1rem;">
            <h3 style="margin-top: 0; color: #990000;">🔴 Priority Screening Recommended</h3>
            <p><strong>Your profile strongly aligns with diabetes risk indicators.</strong></p>
            <p>Strong screening signal detected. Please consult a healthcare provider for proper testing.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- ZONE-AWARE SHAP EXPLANATION ---
    st.subheader("🧠 Key Health Indicators")
    st.markdown("""<span style='color:gray; font-size: 0.9em;'>*Derived from algorithmic feature analysis*</span>""", unsafe_allow_html=True)
    
    # Pass BOTH the dataframe and the probability to trigger the correct tone
    basic_text = generate_text_explanation(df_impact, prob)
    
    st.markdown(f"""
    <div style='background-color: var(--secondary-background-color); padding:20px; border-radius:10px; border-left: 5px solid #5a5a5a;'>
    {basic_text}
    </div>
    """, unsafe_allow_html=True)
    # ... [Keep your existing Basic SHAP Rendering logic above this] ...

    # --- AI ACTION PLAN (Opt-in via Button Click) ---
    st.divider()
    st.subheader("AI driven Insights")
    st.markdown("Get AI driven insights help you get more information")
    
    if st.button("✨ Generate AI Insights"):
        with st.spinner("Analyzing your profile through our AI system..."):
            # Pass the df_impact and zone to our dual-prompt routing engine
            st.session_state.ai_text = generate_ai_insights(df_impact, st.session_state.zone)

    # Render AI Text
    if "ai_text" in st.session_state:
        # Check if it's an error message (quota exhausted on all models)
        if "⚠️" in st.session_state.ai_text or "API Error" in st.session_state.ai_text:
            st.error(st.session_state.ai_text)
        else:
            # Dynamic color logic based on the risk zone
            if st.session_state.zone == "Low Risk":
                border_color = "#4CAF50"  # Green
            elif st.session_state.zone == "Moderate Risk":
                border_color = "#FFEB3B"  # Yellow
            elif st.session_state.zone == "Elevated Risk":
                border_color = "#FF9800"  # Orange
            else: # Priority Screening
                border_color = "#F44336"  # Red
                
            # Clean UI for the AI text with dynamic border color
            st.markdown(f"""
            <div style='background-color: var(--secondary-background-color); padding:20px; border-radius:10px; border-left: 5px solid {border_color}; line-height: 1.6;'>
            {st.session_state.ai_text}
            </div>
            """, unsafe_allow_html=True)

# 8. QUICK GUIDANCE
    st.divider()
    with st.expander("📚 Understanding Your Screening & Next Steps"):
        if st.session_state.zone == "Low Risk":
            st.info("""
            **How to maintain this:**
            * **Routine:** Keep up your current physical activity.
            * **Screening:** Even at low risk, an annual checkup is recommended if you are over 45.
            * **Nutrition:** Focus on a high-fiber, low-sugar diet.
            """)
        elif st.session_state.zone == "Moderate Risk":
            st.warning("""
            **Preventative Actions:**
            * **Active Minutes:** Aim for at least 150 minutes of moderate activity per week.
            * **Weight Management:** Even a 5% weight loss can significantly reduce diabetes risk.
            * **Monitor:** Keep an eye on your blood pressure and cholesterol levels annually.
            """)
        else: # Elevated or Priority
            st.error("""
            **Immediate Next Steps:**
            * **Consultation:** Show these results to your primary care doctor.
            * **Blood Work:** Ask about an A1C or Fasting Plasma Glucose (FPG) test.
            * **Preparation:** Note down any symptoms like unusual thirst, frequent urination, or blurred vision.
            """)