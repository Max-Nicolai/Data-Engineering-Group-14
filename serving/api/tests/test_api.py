"""Tests for the prediction API.

Cloud Build runs these tests after downloading the newest model to
serving/api/model.pkl and before building and deploying the API.
If a test fails, the build stops and the current API stays online.
"""
from main import app  # main.py loads model.pkl when it is imported

# One house from the California Housing dataset (the 8 features the API expects)
SAMPLE_HOUSE = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984,
    "AveBedrms": 1.024,
    "Population": 322.0,
    "AveOccup": 2.556,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


def test_predict_returns_a_price():
    client = app.test_client()
    response = client.post("/predict", json=SAMPLE_HOUSE)

    assert response.status_code == 200
    assert "predicted_price" in response.get_json()


def test_predicted_price_is_realistic():
    # House values in this dataset lie between about $15,000 and $500,001
    client = app.test_client()
    price = client.post("/predict", json=SAMPLE_HOUSE).get_json()["predicted_price"]

    assert 0 < price <= 500001