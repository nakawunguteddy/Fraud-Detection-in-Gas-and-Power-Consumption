import shap
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Load your trained sparse model
model = load_model("model/ann_sparse_model.h5")

# Selected features used during training
SELECTED_FEATURES = ['disrict', 'client_id', 'client_catg', 'region', 'creation_date',
                     'tarif_type', 'counter_number', 'counter_code', 'reading_remarque',
                     'counter_coefficient', 'consommation_level_1', 'counter_type']

# Initialize label encoders
label_encoders = {}

# Load training data for background reference 
X_train_full = pd.read_csv("hybrid_sampled_data.csv")
#X_train_full = pd.read_csv("C:\\Users\\user\\Desktop\\DATASET\\hybrid_sampled_data.csv")  

# Sample 100 rows for SHAP background data
background_data = shap.kmeans(X_train_full[SELECTED_FEATURES], 10)

# Preprocess function for new input
def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    for col in df.columns:
        if df[col].dtype == 'object':
            if col not in label_encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
            else:
                le = label_encoders[col]
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen labels by assigning a fallback value (like -1)
                    df[col] = -1
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep only selected features
    df = df[SELECTED_FEATURES]
    return df

# SHAP explainer function
def get_shap_values(df_input):
    if isinstance(df_input, pd.Series):
        df_input = df_input.to_frame().T
    explainer = shap.KernelExplainer(model.predict, background_data)
    shap_values = explainer.shap_values(df_input)
    return shap_values, explainer


