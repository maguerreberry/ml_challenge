from flask import Flask, jsonify, request
import joblib
import pandas as pd
from titanic.predict import make_prediction
from titanic.config import config

app = Flask(__name__)

@app.route('/', methods=['GET'])
def default():
    return jsonify(rsp='Titanic API')

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    
    X = pd.DataFrame(json_data)

    prediction = make_prediction(input_data=X)

    # Converting to int from int64
    # return jsonify({"prediction": tuple(map(int, prediction))})
    return jsonify(prediction)

if __name__ == '__main__':
    clf = joblib.load(config.PIPELINE_SAVE_FILE)
    app.run(port=5000)
