import json
import pandas as pd
from titanic.config import config as model_config


def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200


def test_prediction_endpoint_returns_prediction(flask_test_client):
    # Given
    # Load the test data from the titanic package
    # This is important as it makes it harder for the test
    # data versions to get confused by not spreading it
    # across packages.
    test_data = pd.read_csv(model_config.TESTING_DATA_FILE)
    post_json = test_data[0:1].to_json(orient='records')

    # When
    response = flask_test_client.post('/predict',
                                      json=json.loads(post_json))

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    assert prediction[0] == 0
