from flask import Flask, jsonify, request
import joblib
import numpy as np

# Load the trained model
model = joblib.load("models/model.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.json['features']
        prediction = model.predict([np.array(data)])

        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/prediction-batch', methods=['POST'])
def prediction_batch():
    try:
        # 1. Extract the 2D list of features from the request body
        data = request.json['features']

        # 2. Convert to numpy array (shape: [n_samples, n_features])
        X = np.array(data)

        # 3. Predict for all samples and cast outputs to native Python ints
        predictions = model.predict(X).tolist()

        # 4. Return as JSON matching the expected response schema: {"predictions": [...]}
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)