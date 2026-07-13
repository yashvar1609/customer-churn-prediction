import os
import pandas as pd
import numpy as np

class TelcoPreprocessor:
    def __init__(self, scaler=None, feature_columns=None):
        self.scaler = scaler
        self.feature_columns = feature_columns
        
        self.binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        self.service_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        self.nominal_cols = [
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
            "Contract", "PaymentMethod", "tenure_bucket"
        ]
        self.numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "num_services"]
        
    def clean_and_engineer(self, df):
        # Create a copy to prevent modifying original
        X = df.copy()
        
        # Drop customerID if present
        if "customerID" in X.columns:
            X = X.drop(columns=["customerID"])
            
        # Clean TotalCharges
        if "TotalCharges" in X.columns:
            # Check if it is not a numeric type (robust to object, pyarrow string, etc.)
            if not pd.api.types.is_numeric_dtype(X["TotalCharges"]):
                X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
            X["TotalCharges"] = X["TotalCharges"].fillna(0.0)
        else:
            X["TotalCharges"] = 0.0
            
        # Engineer tenure_bucket
        if "tenure" in X.columns:
            # Ensure tenure is numeric
            if not pd.api.types.is_numeric_dtype(X["tenure"]):
                X["tenure"] = pd.to_numeric(X["tenure"], errors="coerce").fillna(0)
            X["tenure_bucket"] = pd.cut(
                X["tenure"], 
                bins=[-1, 12, 24, 48, 72], 
                labels=["0-12", "13-24", "25-48", "49-72"]
            )
        else:
            X["tenure"] = 0
            X["tenure_bucket"] = "0-12"
            
        # Engineer avg_monthly_spend
        X["avg_monthly_spend"] = X["TotalCharges"] / (X["tenure"] + 1)
        
        # Engineer num_services
        present_services = [c for c in self.service_cols if c in X.columns]
        if present_services:
            X["num_services"] = (X[present_services] == "Yes").sum(axis=1)
        else:
            X["num_services"] = 0
            
        # Engineer auto_payment
        if "PaymentMethod" in X.columns:
            X["auto_payment"] = X["PaymentMethod"].str.contains("automatic", case=False).fillna(False).astype(int)
        else:
            X["auto_payment"] = 0
            
        # Map binary columns
        if "gender" in X.columns:
            X["gender"] = X["gender"].map({"Female": 1, "Male": 0}).fillna(0).astype(int)
            
        for col in self.binary_cols:
            if col in X.columns:
                X[col] = X[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
                
        # Handle target variable Churn if present
        if "Churn" in X.columns:
            X["Churn"] = X["Churn"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
            
        return X

    def transform_inference(self, raw_df):
        # 1. Clean and engineer
        X = self.clean_and_engineer(raw_df)
        
        # 2. One-hot encode multi-category columns
        X_encoded = pd.get_dummies(X, columns=self.nominal_cols, drop_first=True, dtype=int)
        
        # 3. Align with feature_columns (reindex and fill with 0)
        if self.feature_columns is not None:
            # Drop Churn if it got encoded or exists
            if "Churn" in X_encoded.columns:
                X_encoded = X_encoded.drop(columns=["Churn"])
            X_encoded = X_encoded.reindex(columns=self.feature_columns, fill_value=0)
            
        # 4. Scale numeric features
        if self.scaler is not None:
            X_encoded[self.numeric_cols] = self.scaler.transform(X_encoded[self.numeric_cols])
            
        return X_encoded
