import pandas as pd
import joblib
import logging
from titanic import pipeline
from titanic.config import config

_logger = logging.getLogger(__name__)

def run_training() -> None:
    """Train the model."""

    # read training data
    data = pd.read_csv(config.TRAINING_DATA_FILE)

    pipeline.titanic_pipe.fit(data[config.FEATURES], data[config.TARGET])
    joblib.dump(pipeline.titanic_pipe, config.PIPELINE_SAVE_FILE)
    _logger.info(f"saving model titanic")

if __name__ == '__main__':
    run_training()
