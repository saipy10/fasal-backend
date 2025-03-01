import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define Flask app
app = Flask(__name__)

# Encoders (must match training phase)
encoder_commodity = {
    "Paddy": 0,
    "Jowar": 1,
    "Bajra": 2,
    "Ragi": 3,
    "Maize": 4,
    "Tur (Arhar)": 5,
    "Moong": 6,
    "Urad": 7,
}

encoder_state = {
    "West Bengal": 0,
    "Maharashtra": 1,
    "Rajasthan": 2,
    "Karnataka": 3,
    "Madhya Pradesh": 4,
}

# Feature order (must match training)
feature_order = [
    "Commodity",
    "2013-14", "2014-15", "2015-16", "2016-17", "2017-18",
    "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
    "State", "Annual Rainfall (mm)"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input JSON
        data = request.get_json()

        # Extract features
        commodity = data.get("Commodity")
        state = data.get("State")
        annual_rainfall = data.get("Annual Rainfall")

        # Validate inputs
        if not commodity or not state or annual_rainfall is None:
            return jsonify({"error": "Missing required fields"}), 400

        if commodity not in encoder_commodity or state not in encoder_state:
            return jsonify({"error": "Invalid Commodity or State"}), 400

        # Convert and encode categorical values
        commodity_encoded = encoder_commodity[commodity]
        state_encoded = encoder_state[state]
        annual_rainfall = float(annual_rainfall)

        # Use the mean of past prices for missing historical data
        avg_past_price = np.mean([
            2300, 2320, 3371, 3421, 2625, 4290, 2225, 7550, 8682, 7400
        ])

        # Construct input in the **correct feature order**
        input_data = pd.DataFrame(
            [[commodity_encoded] + [avg_past_price] * 11 + [state_encoded, annual_rainfall]],
            columns=feature_order
        )

        # Predict price and convert to standard float
        predicted_price = float(model.predict(input_data)[0])

        return jsonify({"Predicted Crop Price": round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
