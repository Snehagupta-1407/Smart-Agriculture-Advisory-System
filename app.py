import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load trained ML model
model = joblib.load('smart-agriculture.joblib')

# Fertilizer Recommendation Dictionary
fertilizer_dic = {
    "rice": "Urea and DAP",
    "wheat": "Nitrogen Rich Fertilizer",
    "maize": "NPK Fertilizer",
    "cotton": "Potassium Fertilizer",
    "coffee": "Organic Compost",
    "banana": "Phosphorus Fertilizer",
    "mango": "Compost and Potash",
    "apple": "Nitrogen Fertilizer",
    "grapes": "Organic Fertilizer",
    "lentil": "DAP and Compost"
}

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get JSON data from frontend
        data = request.json

        # Extract features
        features = [
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]

        # Convert into numpy array
        features = np.array(features).reshape(1, -1)

        # Predict crop
        prediction = model.predict(features)[0]

        # Convert prediction to lowercase
        crop_name = str(prediction).lower()

        # Get fertilizer recommendation
        fertilizer = fertilizer_dic.get(
            crop_name,
            "General Organic Fertilizer"
        )

        # Return response
        return jsonify({
            'crop': prediction,
            'fertilizer': fertilizer
        })


    except Exception as e:

        return jsonify({
            'error': str(e)
        })

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)