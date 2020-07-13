import pandas as pd
from titanic.predict import make_prediction
from titanic.config import config


def test_make_single_prediction():
    # Given
    test_data = pd.read_csv(config.TESTING_DATA_FILE)
    single_test_input = test_data[0:1]

    # When
    subject = make_prediction(input_data=single_test_input)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], int)
    assert subject.get('predictions')[0] == 0


def test_make_multiple_predictions():
    # Given
    test_data = pd.read_csv(config.TESTING_DATA_FILE)
    original_data_length = len(test_data)
    multiple_test_input = test_data[config.FEATURES]

    # When
    subject = make_prediction(input_data=multiple_test_input)

    # Then
    assert subject is not None
    assert len(subject.get('predictions')) == 418
