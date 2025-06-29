import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='template')

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    if not model:
        return render_template("output.html", result="Model not loaded. Please check model.pkl")

    try:
        input_features = [float(x) for x in request.form.values()]
        feature_names = ['holiday', 'temp', 'rain', 'snow', 'weather',
                         'year', 'month', 'day', 'hours', 'minutes', 'seconds']
        data = pd.DataFrame([input_features], columns=feature_names)
        prediction = model.predict(data)[0]
        result_text = f"Estimated Traffic Volume is: {int(prediction)} units"
        return render_template("output.html", result=result_text)
    except Exception as e:
        return render_template("output.html", result=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
