import os
from flask import Flask, render_template, request
import requests

app = Flask(__name__)
API_URL = os.environ.get("PREDICTOR_API", "http://localhost:5000/predict")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    payload = {key: float(val) for key, val in request.form.items()}
    res = requests.post(API_URL, json=payload)
    prediction = res.json().get("predicted_price", "Error")
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
