from flask import Flask, render_template, request
import numpy as np
from utils import preprocess_input, get_shap_values, model
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fallback_key")


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    shap_summary = None
    if request.method == 'POST':
        user_input = {key: request.form[key] for key in request.form}
        df_input = preprocess_input(user_input)
        pred_prob = model.predict(df_input)[0][0]
        prediction = "Fraud" if pred_prob >= 0.5 else "Not Fraud"

        shap_values, explainer = get_shap_values(df_input)
        shap_summary = shap_values[0]  

    return render_template('index.html', prediction=prediction, shap_summary=shap_summary)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

