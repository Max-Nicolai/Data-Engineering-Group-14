# data/train.py
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=20, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("Baseline model.pkl created successfully.")
