import pathlib
import titanic

# data
PACKAGE_ROOT = pathlib.Path(titanic.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINING_DATA_FILE = DATASET_DIR / 'train.csv'
TESTING_DATA_FILE = DATASET_DIR / 'test.csv'
PIPELINE_SAVE_FILE = TRAINED_MODEL_DIR / 'titanic_model.pkl'
PIPELINE_SAVE_FILE_B = TRAINED_MODEL_DIR / 'titanic_model_b.pkl'

TARGET = 'Survived'

# input variables
FEATURES = ['Age', 'Sex', 'Embarked', 'Pclass']

CATEGORICAL_FEATURES = ['Sex', 'Embarked', 'Pclass']

NUMERICAL_FEATURES = ['Age']
