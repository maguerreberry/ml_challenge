import joblib
import logging
import pandas as pd
import numpy as np
from titanic.config import config

_logger = logging.getLogger(__name__)
_pipe_titanic = joblib.load(filename=config.PIPELINE_SAVE_FILE)
_pipe_titanic_b = joblib.load(filename=config.PIPELINE_SAVE_FILE_B)


def make_prediction(*, input_data: pd.DataFrame) -> dict:
    """
    Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row.
    """

    model = {
        0: {
            'name': 'RandomForest',
            'pipe': _pipe_titanic,
        },
        1: {
            'name': 'Logistic',
            'pipe': _pipe_titanic_b,
        },
    }

    r = np.random.randint(low=0, high=2, size=1)[0]

    input_data = input_data.reindex(columns=config.FEATURES)

    predictions = model[r]['pipe'].predict(input_data[config.FEATURES])

    results =  {
        'predictions': tuple(map(int, predictions)),
        'model': model[r]['name'],
    }

    _logger.info(
        f"\nInputs:\n{input_data}"
        f"\nPredictions:\n{results}"
    )

    return results

if __name__ == '__main__':

    # test pipeline
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    data = pd.read_csv(config.TESTING_DATA_FILE)

    X_test = data[config.FEATURES]

    pred = make_prediction(input_data=X_test)
