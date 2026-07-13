import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import shap

# Append src/ directory to path so we can import our preprocessor
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import TelcoPreprocessor

# Set page config
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #1e3d59;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #17b978;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #118f5c;
        color: white;
    }
    .report-card {
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .low-risk {
        background-color: #28a745;
    }
    .med-risk {
        background-color: #ffc107;
        color: #212529;
    }
    .high-risk {
        background-color: #dc3545;
    }
    .rec-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #17b978;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# 1. Load serialized models and preprocessing assets
@st.cache_resource
def load_assets():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model = joblib.load(os.path.join(models_dir, "best_churn_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(models_dir, "model_features.pkl"))
    return model, scaler, feature_columns

# 2. Cache SHAP explainer at startup
@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

# Load assets
try:
    model, scaler, feature_columns = load_assets()
    explainer = get_shap_explainer(model)
    preprocessor = TelcoPreprocessor(scaler=scaler, feature_columns=feature_columns)
except Exception as e:
    st.error(f"Error loading model assets: {e}. Please run 'python src/train.py' first to serialize models.")
    st.stop()

# Header
st.title("📊 Telco Customer Churn Risk Assessment")
st.markdown("Assess customer churn probability using our tuned machine learning classifier. Provides **explainable AI (SHAP)** visualizations and **tailored business retention recommendations** for individual customer accounts.")
st.markdown("---")

# Layout columns
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.subheader("👤 Customer Profile Input Form")
    
    with st.form("customer_input_form"):
        # Form Sections
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        with c2:
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12, step=1)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", 
                "Mailed check", 
                "Bank transfer (automatic)", 
                "Credit card (automatic)"
            ])
        with c3:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=250.0, value=70.0, step=0.5)
            # Help user estimate total charges if they don't know it
            total_charges_est = tenure * monthly_charges
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=total_charges_est, step=1.0)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            
        st.markdown("**Subscribed Services:**")
        s1, s2, s3 = st.columns(3)
        with s1:
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        with s2:
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        with s3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submit_btn = st.form_submit_button("Predict Churn Risk")

with col2:
    st.subheader("📈 Assessment & Recommendations")
    
    if submit_btn:
        # 1. Gather form input into a DataFrame
        raw_customer_dict = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        
        df_customer = pd.DataFrame([raw_customer_dict])
        
        # 2. Preprocess raw data (clean, engineer, encode, scale)
        df_processed = preprocessor.transform_inference(df_customer)
        
        # 3. Predict Churn Probability
        prob = model.predict_proba(df_processed)[0, 1]
        prob_pct = prob * 100
        
        # Risk Classification & Reporting Card
        if prob_pct < 30.0:
            risk_class = "low-risk"
            risk_label = "LOW RISK"
            risk_desc = "This customer shows high loyalty signals. Keep engaging with standard updates."
        elif prob_pct <= 60.0:
            risk_class = "med-risk"
            risk_label = "MEDIUM RISK"
            risk_desc = "This customer shows borderline warning signs. Target with preventive loyalty offers."
        else:
            risk_class = "high-risk"
            risk_label = "HIGH RISK"
            risk_desc = "This customer is highly likely to churn. Immediate retention intervention required!"
            
        st.markdown(f"""
        <div class="report-card {risk_class}">
            <h3 style="margin:0; color:inherit;">{risk_label}</h3>
            <h1 style="margin:0; font-size: 3rem; color:inherit;">{prob_pct:.1f}%</h1>
            <p style="margin:5px 0 0 0; color:inherit;">Churn Probability</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(risk_desc)
        
        # 4. Generate SHAP Explanations
        st.markdown("#### 🔍 Explainable AI (Why is this customer flagged?)")
        
        with st.spinner("Generating SHAP waterfall explanation..."):
            shap_values = explainer.shap_values(df_processed)
            
            # Construct explanation object
            indiv_explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=df_processed.values[0],
                feature_names=feature_columns
            )
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(indiv_explanation, max_display=10, show=False)
            plt.title(f"SHAP Contribution to Churn Risk", fontsize=11, pad=10)
            st.pyplot(fig)
            plt.close()
            
        # 5. Tailored Business Recommendations
        st.markdown("#### 📋 Actionable Retention Playbook")
        
        recommendations = []
        
        # Rule 1: Contract Upgrade
        if contract == "Month-to-month":
            recommendations.append(
                "**Upgrade to Long-Term Contract:** Offer a contract transition campaign. Transitioning this customer to a 1-year contract (even with a small \$5/month discount) directly pulls down churn probability by securing contract loyalty."
            )
            
        # Rule 2: Payment Method transition
        if payment_method == "Electronic check":
            recommendations.append(
                "**Incentivize Autopay Enrollment:** Electronic check has manual payment friction. Offer a one-time \$10 bill credit to enroll this customer in automated credit card or bank transfer billing."
            )
            
        # Rule 3: Fiber Optic satisfaction audit
        if internet_service == "Fiber optic":
            recommendations.append(
                "**Fiber Optic Service Audit:** Fiber optic customers show high pricing complaints and technical churn. Trigger a proactive technical health check on their router and review their billing plan."
            )
            
        # Rule 4: Service add-ons push
        num_services = (df_customer[preprocessor.service_cols] == "Yes").sum(axis=1).values[0]
        if num_services == 0:
            recommendations.append(
                "**Bundle Service Add-ons:** This customer has zero value-added services. Pitch a discounted security bundle (OnlineSecurity + TechSupport) for \$5/month. Adding services increases churn friction."
            )
            
        # Default rule
        if not recommendations:
            recommendations.append(
                "**Loyalty Appreciation:** Customer is in a highly secure profile. Maintain customer relationship and offer early access to new products or hardware upgrades."
            )
            
        # Render recommendations
        for rec in recommendations:
            st.markdown(f"""
            <div class="rec-box">
                {rec}
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.write("👈 Fill in the customer profile details on the left and click **'Predict Churn Risk'** to perform the assessment.")
