import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    # 8 features in exact order
    features = [
        float(data["MedInc"]),
        float(data["HouseAge"]),
        float(data["AveRooms"]),
        float(data["AveBedrms"]),
        float(data["Population"]),
        float(data["AveOccup"]),
        float(data["Latitude"]),
        float(data["Longitude"]),
    ]
    pred = model.predict([features])[0]
    # Target is in hundreds of thousands of dollars ($100,000s)
    return jsonify({"predicted_price": round(float(pred * 100000), 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
