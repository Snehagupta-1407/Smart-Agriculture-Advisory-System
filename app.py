from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from dotenv import load_dotenv
import requests
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
load_dotenv()
API_KEY = os.getenv("API_KEY")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print(API_KEY)
app = Flask(__name__)

# Load trained ML model
model = joblib.load('crop_model.joblib')
disease_model = load_model('plant_disease_model.h5')
class_names = [

    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',

    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy',

    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',

    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___healthy'
]

disease_solutions = {

    "Potato___Early_blight":
    "Apply Mancozeb fungicide every 7-10 days and remove infected leaves.",

    "Potato___Late_blight":
    "Apply Chlorothalonil fungicide and avoid excessive moisture.",

    "Tomato___Bacterial_spot":
    "Use copper-based bactericides and remove infected leaves.",

    "Tomato___Early_blight":
    "Apply fungicide and improve air circulation around plants.",

    "Tomato___Late_blight":
    "Avoid overhead watering and use preventive fungicides.",

    "Apple___Apple_scab":
    "Use sulfur-based fungicides and remove fallen leaves.",

    "Apple___Black_rot":
    "Prune infected branches and apply fungicides.",

    "Corn_(maize)___Common_rust_":
    "Use resistant hybrids and recommended fungicides.",

    "Potato___healthy":
    "No disease detected. Continue regular crop care.",

    "Tomato___healthy":
    "Healthy plant. Maintain proper irrigation and nutrition.",

    "Apple___healthy":
    "Healthy plant. Continue regular monitoring."
}

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
# Season Recommendation Dictionary
season_dic = {

    "rice": "Kharif",

    "wheat": "Rabi",

    "maize": "Kharif",

    "cotton": "Kharif",

    "coffee": "Whole Year",

    "banana": "Whole Year",

    "mango": "Summer",

    "apple": "Winter",

    "grapes": "Summer",

    "lentil": "Rabi",

    "orange": "Winter",

    "watermelon": "Summer"

}
# Irrigation Recommendation Dictionary
irrigation_dic = {

    "rice": "Flood Irrigation",

    "wheat": "Sprinkler Irrigation",

    "maize": "Drip Irrigation",

    "cotton": "Drip Irrigation",

    "banana": "Drip Irrigation",

    "mango": "Basin Irrigation",

    "apple": "Sprinkler Irrigation",

    "grapes": "Drip Irrigation",

    "orange": "Drip Irrigation",

    "watermelon": "Drip Irrigation"
}
# Home Route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/weather')
def weather():

    city = request.args.get('city')

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)

    data = response.json()
    print(data)

    if data.get("cod") != 200:

        return jsonify({
            "error": "City not found"
        })

    temperature = data['main']['temp']

    humidity = data['main']['humidity']

    return jsonify({
        "temperature": temperature,
        "humidity": humidity
    })
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

        # Get soil type
        soil = data.get('soil')

        # Fertilizer recommendation
        fertilizer = "General Fertilizer"

        if soil == "Black Soil":

            fertilizer = "Urea and Compost"

        elif soil == "Red Soil":

            fertilizer = "Phosphorus Fertilizer"

        elif soil == "Clay Soil":

            fertilizer = "Organic Fertilizer"

        elif soil == "Alluvial Soil":

            fertilizer = "Nitrogen Fertilizer"

        elif soil == "Sandy Soil":

            fertilizer = "Potassium Fertilizer"

       
        # Get season recommendation
        season = season_dic.get(
            crop_name,
            "Suitable in All Seasons"
        )
        # Get irrigation recommendation
        irrigation = irrigation_dic.get(

           crop_name,

            "Standard Irrigation"
        )
        # Soil Health Status

        soil_health = "Moderate"

        if (
            data['N'] > 50 and
            data['P'] > 40 and
            data['K'] > 40 and
            5.5 <= data['ph'] <= 7.5
        ):

            soil_health = "Healthy"

        elif (
           data['N'] < 20 or
           data['P'] < 20 or
           data['K'] < 20
        ):

         soil_health = "Poor" 

        # Return response
        return jsonify({

            'crop': prediction,

            'fertilizer': fertilizer,

            'season': season,

            'irrigation': irrigation,

            'soil_health': soil_health
        })

    except Exception as e:

        return jsonify({
            'error': str(e)
        })
    

@app.route('/detect-disease', methods=['POST'])
def detect_disease():

    try:

        file = request.files['image']

        os.makedirs("uploads", exist_ok=True)

        filepath = os.path.join("uploads", file.filename)

        file.save(filepath)

        # Load image
        img = image.load_img(
            filepath,
            target_size=(128, 128)
        )

        img_array = image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)

        img_array = img_array / 255.0

        # Prediction
        prediction = disease_model.predict(img_array)

        predicted_class = class_names[np.argmax(prediction)]

        confidence = float(np.max(prediction)) * 100

        solution = disease_solutions.get(
            predicted_class,w
            "Consult an agriculture expert."
        )

        return jsonify({

            "disease": predicted_class,

            "confidence": round(confidence, 2),

            "solution": solution
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

@app.route('/chat', methods=['POST'])
def chat():

    try:

        data = request.json

        user_message = data['message']

        headers = {

            "Authorization": f"Bearer {OPENROUTER_API_KEY}",

            "Content-Type": "application/json"
        }

        payload = {

    "model": "openai/gpt-3.5-turbo",

    "messages": [

        {
            "role": "system",

            "content": """
            You are an expert AI agriculture assistant.

            Help users with:
            - farming
            - crops
            - fertilizers
            - irrigation
            - soil health
            - weather
            - diseases
            - agriculture technology
            """
        },

        {
            "role": "user",

            "content": user_message
        }
    ]
}

        response = requests.post(

            url="https://openrouter.ai/api/v1/chat/completions",

            headers=headers,

            json=payload
        )

        result = response.json()

        print(result)

        if 'choices' in result:

           ai_reply = result['choices'][0]['message']['content']

        else:

            ai_reply = str(result)
        return jsonify({
            'reply': ai_reply
        })

    except Exception as e:

        return jsonify({
            'reply': str(e)
        })



# Run Flask App
if __name__ == '__main__':
    app.run(debug=True, port=8000)

    