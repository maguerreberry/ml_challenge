import pandas as pd
from titanic.predict import make_prediction
from titanic.config import config


def test_make_single_prediction():
    # Given
    # test_data = pd.read_csv(config.TESTING_DATA_FILE)
    # single_test_input = test_data[0:1][config.FEATURES]
    single_test_input = pd.DataFrame([
        {"Age": 85, "Sex": "male",   "Embarked": "S", "Pclass": 1},
        {"Age": 24, "Sex": "female", "Embarked": "C", "Pclass": 1},
        {"Age": 3,  "Sex": "male",   "Embarked": "C", "Pclass": 1},
        {"Age": 44, "Sex": "female", "Embarked": "Q", "Pclass": 1},
        {"Age": 21, "Sex": "male",   "Embarked": "S", "Pclass": 1}
    ])

    # When
    subject = make_prediction(input_data=single_test_input)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], int)
    assert subject.get('predictions')[0] == 1


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
