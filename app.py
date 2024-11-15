from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict_sleep_efficiency():
    try:
        # Get input data from JSON request
        data = request.json

        # Example: Ensure the expected input format
        required_fields = ['timetobed', 'timeoutofbed', 'bedtimedur',
       'complypercent', 'meanrate', 'sdrate', 'steps', 'floors',
       'sedentaryminutes', 'lightlyactiveminutes', 'fairlyactiveminutes',
       'veryactiveminutes', 'lowrangemins', 'fatburnmins', 'cardiomins',
       'peakmins', 'lowrangecal', 'fatburncal', 'cardiocal', 'peakcal',
       'gender', 'caffine_consumption', 'happiness_level',
       'alcohol_consumption']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Prepare input data for the model
        input_features = np.array([[
            data['timetobed'],
            data['timeoutofbed'],
            data['bedtimedur'],
            data['complypercent'],
            data['meanrate'],
            data['sdrate'],
            data['steps'],
            data['floors'],
            data['sedentaryminutes'],
            data['lightlyactiveminutes'],
            data['fairlyactiveminutes'],
            data['veryactiveminutes'],
            data['lowrangemins'],
            data['fatburnmins'],
            data['cardiomins'],
            data['peakmins'],
            data['lowrangecal'],
            data['fatburncal'],
            data['cardiocal'],
            data['peakcal'],
            data['gender'],
            data['caffine_consumption'],
            data['happiness_level'],
            data['alcohol_consumption']
        ]])

        # Scale the input features
        input_features_scaled = scaler.transform(input_features)

        # Predict using the trained model
        prediction = model.predict(input_features_scaled)

        # Return prediction as JSON
        return jsonify({"sleep_efficiency_class": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
