# Telco Customer Churn Prediction

A Streamlit application that estimates telecom customer churn risk from a customer profile. It uses a saved XGBoost classifier, explains an individual prediction with SHAP, and displays rule-based retention suggestions.

## What the project does

- Accepts customer, account, billing, and subscribed-service details in a Streamlit form.
- Cleans and engineers features using the same preprocessing logic used for training.
- Aligns one-hot encoded input with the saved model feature schema and scales numeric columns.
- Displays a churn probability and a simple risk band:
  - **Low:** below 30%
  - **Medium:** 30% through 60%
  - **High:** above 60%
- Displays a SHAP waterfall explanation for the individual prediction.
- Suggests retention actions based on the entered contract, payment method, internet service, and service add-ons.

The risk bands and retention suggestions are application rules. They are not calibrated business decisions or proof that a suggested action will change a customer's behavior.

## Repository contents

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── best_churn_model.pkl
│   ├── model_features.pkl
│   └── scaler.pkl
├── src/
│   ├── preprocessing.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

The `models/` files are pre-trained artifacts used by the app. The `data/` CSV is used if you choose to retrain. Local virtual environments, Python caches, and editor files are excluded through `.gitignore`; do not commit a `.venv/` folder.

## Requirements

- Python 3.11 recommended.
- `pip`
- Internet access when installing the packages listed in `requirements.txt`.

The application dependencies are listed in [`requirements.txt`](requirements.txt). No notebook or separate database setup is required to launch the saved model.

## Run locally

Run these commands from the repository root (the directory containing this README).

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

If PowerShell prevents environment activation, call the environment executables directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) when the application starts. Keep the terminal running while you use the app.

## Retrain the model

The checked-in model artifacts allow you to run the app without training first. To regenerate them from the dataset, activate the environment and run this command from the repository root:

```bash
python src/train.py
```

The training script reads `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`, preprocesses the data and drops duplicate processed rows, creates a stratified 80/20 train/test split using `random_state=42`, fits a `StandardScaler` on the training portion, trains an XGBoost classifier, prints accuracy, recall, F1, and ROC-AUC for the held-out portion, and writes the model artifacts to `models/`.

The selected model settings in the training script are `n_estimators=100`, `max_depth=3`, `learning_rate=0.05`, `subsample=0.6`, and `colsample_bytree=0.8`; the positive-class weight is calculated from the training labels. Training overwrites the model files currently used by the app.

Previously reported held-out results for this configuration were accuracy **0.7452**, recall **0.7930**, F1 **0.6224**, and ROC-AUC **0.8427**. Retraining prints fresh scores; results can differ with library versions or changes to the data.

## Preprocessing and model files

`src/preprocessing.py` defines `TelcoPreprocessor`. It:

1. Drops `customerID` when present and converts invalid or blank `TotalCharges` values to zero.
2. Derives `tenure_bucket` (0-12, 13-24, 25-48, and 49-72 months), `avg_monthly_spend`, `num_services` (count of six internet add-on services), and `auto_payment`.
3. Maps binary values and one-hot encodes nominal features such as contract, internet service, payment method, and tenure bucket.
4. Aligns inference columns to the saved feature list and scales `tenure`, `MonthlyCharges`, `TotalCharges`, `avg_monthly_spend`, and `num_services` with the saved scaler.

The model files have separate roles:

- `models/best_churn_model.pkl` — fitted XGBoost classifier.
- `models/model_features.pkl` — feature names and order expected by the classifier.
- `models/scaler.pkl` — fitted numeric-feature scaler.

Keep these files together. If you retrain the model, regenerate all three together so their features and scaling stay consistent.

## Dataset

The included CSV is the IBM Telco Customer Churn dataset, commonly distributed through [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It contains 7,043 customer records and a `Churn` target (`Yes` or `No`); the project data has 1,869 churned and 5,174 retained customers. The source data includes 11 blank `TotalCharges` entries, which the preprocessing code treats as zero.

Please check and follow the dataset provider's terms and attribution requirements when redistributing the CSV.

## Limitations and responsible use

- This is a demonstration project, not a validated production decision system. Validate model quality, thresholds, fairness, and business impact with representative current data before using results operationally.
- The training data covers tenure up to 72 months and the engineered tenure buckets end at 72 months. Although the form permits entering a higher value, predictions outside the training range are not validated.
- The probability estimates and rule-based suggestions are based on the saved model and entered profile; the app does not connect to a telecom provider's customer database.
- Do not enter real customer personal or confidential information into a public deployment unless you have an approved privacy and security design.
- The model and scaler are serialized with Joblib. Only load trusted model files; loading untrusted pickle-based files can execute code.
- The repository does not currently include a `LICENSE` file. Add an appropriate license before granting others reuse or redistribution rights.

## Acknowledgments

The project uses the IBM Telco Customer Churn dataset and the Python libraries listed in [`requirements.txt`](requirements.txt), including Streamlit, pandas, scikit-learn, XGBoost, SHAP, and Joblib.
