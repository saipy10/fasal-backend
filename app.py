import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Encoders (unchanged)
encoder_commodity = {
    "Wheat": 0, "Rice": 1, "Maize": 2, "Paddy": 3, "Bajra": 4, "Mustard": 5,
    "Ragi": 6, "Tur (Arhar)": 7, "Jowar": 8, "Groundnut": 9, "Cotton": 10,
    "Soybean": 11, "Sugarcane": 12, "Urad": 13, "Sunflower": 14, "Moong": 15
}

encoder_state = {
    "Punjab": 0, "Haryana": 1, "Karnataka": 2, "Andhra Pradesh": 3, "Telangana": 4,
    "Maharashtra": 5, "Madhya Pradesh": 6, "Tamil Nadu": 7, "Gujarat": 8,
    "Uttar Pradesh": 9, "Rajasthan": 10
}

# Feature order (unchanged)
feature_order = [
    "Commodity", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18",
    "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
    "State", "Annual Rainfall (mm)"
]

# Mock historical prices per commodity (updated to 11 years: 2013-14 to 2023-24)
historical_prices = {
    "Wheat": [1350, 1400, 1450, 1525, 1625, 1735, 1840, 1925, 2015, 2125, 2275],
    "Rice": [2500, 2600, 2750, 2900, 3100, 3300, 3500, 3700, 3900, 4100, 4350],
    "Maize": [1310, 1360, 1400, 1450, 1520, 1600, 1700, 1800, 1900, 2000, 2225],
    "Paddy": [1310, 1360, 1410, 1470, 1550, 1750, 1815, 1868, 1940, 2040, 2183],
    "Bajra": [1250, 1300, 1350, 1400, 1475, 1550, 1650, 1750, 1850, 1950, 2150],
    "Mustard": [2500, 2600, 2700, 2800, 2950, 3100, 3250, 3400, 3550, 3700, 3850],
    "Ragi": [1500, 1550, 1650, 1725, 1900, 2100, 2500, 2897, 3150, 3378, 3846],
    "Tur (Arhar)": [4300, 4350, 4625, 5050, 5450, 5675, 5800, 6000, 6300, 6600, 7000],
    "Jowar": [1550, 1575, 1625, 1710, 1850, 2100, 2200, 2350, 2550, 2750, 2975],
    "Groundnut": [4000, 4100, 4200, 4300, 4450, 4550, 4650, 4750, 4850, 4950, 5050],
    "Cotton": [3700, 3750, 3800, 3860, 4020, 4220, 4420, 4620, 4820, 5020, 5220],
    "Soybean": [2600, 2750, 2900, 3050, 3390, 3600, 3750, 3880, 3950, 4300, 4600],
    "Sugarcane": [210, 220, 230, 240, 255, 275, 285, 295, 305, 315, 325],
    "Urad": [4000, 4100, 4350, 4625, 5050, 5450, 5625, 5800, 6000, 6300, 6600],
    "Sunflower": [3600, 3700, 3850, 4000, 4340, 4510, 4700, 4880, 5100, 5400, 5650],
    "Moong": [4500, 4600, 4850, 5225, 5575, 5800, 6200, 6975, 7196, 7755, 8558]
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input JSON
        data = request.get_json()

        # Validate JSON format
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input format. Expected JSON."}), 400

        # Extract and validate inputs
        commodity = data.get("Commodity")
        state = data.get("State")
        annual_rainfall = data.get("Annual Rainfall")

        if not commodity or not state or annual_rainfall is None:
            return jsonify({"error": "Missing required fields (Commodity, State, Annual Rainfall)"}), 400

        if commodity not in encoder_commodity:
            return jsonify({"error": f"Invalid Commodity: {commodity}"}), 400
        if state not in encoder_state:
            return jsonify({"error": f"Invalid State: {state}"}), 400

        try:
            annual_rainfall = float(annual_rainfall)
            if annual_rainfall < 0:
                return jsonify({"error": "Annual Rainfall cannot be negative"}), 400
        except ValueError:
            return jsonify({"error": "Annual Rainfall must be a numeric value"}), 400

        # Encode categorical values
        commodity_encoded = encoder_commodity[commodity]
        state_encoded = encoder_state[state]

        # Handle missing historical prices dynamically
        past_prices = historical_prices.get(commodity, [])
        if not past_prices:
            return jsonify({"error": f"Historical data not available for {commodity}"}), 500

        # Ensure past_prices matches the expected length (11 years: 2013-14 to 2023-24)
        if len(past_prices) != 11:
            return jsonify({"error": f"Insufficient historical data for {commodity}"}), 500

        # Construct input data in the correct feature order
        input_data = pd.DataFrame(
            [[commodity_encoded] + past_prices + [state_encoded, annual_rainfall]],
            columns=feature_order
        )

        # Predict price and convert to float
        predicted_price = float(model.predict(input_data)[0])

        logging.info(f"Prediction successful for {commodity} in {state}: ₹{round(predicted_price, 2)}")

        return jsonify({"Predicted Crop Price (₹)": round(predicted_price, 2)})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)