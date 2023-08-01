from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return 'Test message. The server is running'

@app.route('/predict', methods=['POST'])
def predict():
    # features = request.json.get('features')
    features = request.json
    features = np.array(features).reshape(1, 4)
    prediction = round(loaded_model.predict(features)[0], 2)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    with open('data/model.pkl', 'rb') as pkl_file:
        loaded_model = pickle.load(pkl_file)

    app.run('localhost', 5000)
