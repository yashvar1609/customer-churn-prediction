import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from preprocessing import TelcoPreprocessor

def main():
    print("Starting training pipeline...")
    
    # 1. Load raw dataset
    raw_data_path = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_data_path}")
        
    df_raw = pd.read_csv(raw_data_path)
    print(f"Loaded raw dataset of shape: {df_raw.shape}")
    
    # Initialize Preprocessor
    preprocessor = TelcoPreprocessor()
    
    # 2. Clean and engineer features
    df_cleaned = preprocessor.clean_and_engineer(df_raw)
    
    # Drop duplicates in feature space if any exist
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Separate features and target
    X = df_cleaned.drop(columns=["Churn"])
    y = df_cleaned["Churn"]
    
    # 3. One-Hot Encode multi-category columns
    X_encoded = pd.get_dummies(X, columns=preprocessor.nominal_cols, drop_first=True, dtype=int)
    
    # Save the exact list of final encoded feature columns (crucial for reindexing during inference)
    feature_columns = X_encoded.columns.tolist()
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(feature_columns, os.path.join(models_dir, "model_features.pkl"))
    print(f"Saved {len(feature_columns)} feature columns to models/model_features.pkl")
    
    # 4. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Fit and Save StandardScaler (strictly on training set numeric features)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[preprocessor.numeric_cols] = scaler.fit_transform(X_train[preprocessor.numeric_cols])
    X_test_scaled[preprocessor.numeric_cols] = scaler.transform(X_test[preprocessor.numeric_cols])
    
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    print("Saved fitted StandardScaler to models/scaler.pkl")
    
    # Calculate scale_pos_weight for imbalance handling
    ratio = sum(y_train == 0) / sum(y_train == 1)
    print(f"XGBoost scale_pos_weight ratio: {ratio:.4f}")
    
    # 6. Initialize and Fit the final Tuned XGBoost Model
    best_params = {
        'subsample': 0.6,
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.05,
        'colsample_bytree': 0.8
    }
    
    print("Training final Tuned XGBoost model...")
    final_model = XGBClassifier(
        scale_pos_weight=ratio,
        eval_metric='logloss',
        random_state=42,
        **best_params
    )
    
    final_model.fit(X_train_scaled, y_train)
    
    # 7. Evaluate on Test Set to verify consistency
    y_pred = final_model.predict(X_test_scaled)
    y_prob = final_model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print("\nFinal Test Set Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    # Save the trained model
    joblib.dump(final_model, os.path.join(models_dir, "best_churn_model.pkl"))
    print("Saved final model to models/best_churn_model.pkl")
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
