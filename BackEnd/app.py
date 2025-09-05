from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)
# model = joblib.load("X_G_Boost.jb") 
model = joblib.load(r"d:\.aORG Study\Level 2\AIU NOW\Machine Learning [AIE121]\The project\ML Project\BackEnd\X_G_Boost.jb")

# encoder = joblib.load("label_encoder.jb")
encoder = joblib.load(r"d:\.aORG Study\Level 2\AIU NOW\Machine Learning [AIE121]\The project\ML Project\BackEnd\label_encoder.jb")

# print(encoder['category'].classes_)
# print("Gender classes:", encoder['gender'].classes_)
def preprocess_input(data):
    # ترميز category و gender
    category_encoded = encoder['category'].transform([data['category']])[0]
    gender_encoded = encoder['gender'].transform([data['gender']])[0]
    features = [
        data['cc_num'],         # Card Number
        category_encoded,       # Transaction Category 
        data['amt'],            # Transaction Amount
        gender_encoded,         # Gender 
        data['zip'],            # ZIP code
        data['lat'],            # Latitude
        data['long'],           # Longitude
        data['merch_lat'],      # Merchant Latitude
        data['merch_long'],     # Merchant Longitude
        data['hour'],           # Hour
        data['day'],            # Day
        data['month'],          # Month
        data['distance']        # Distance
    ]
    return np.array([features])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = preprocess_input(data)
    prediction = model.predict(X)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)