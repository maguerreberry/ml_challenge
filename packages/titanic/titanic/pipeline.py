from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import titanic.preprocessors as pp
from titanic.config import config


numeric_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ]
)

categorical_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ]
)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_transformer, config.NUMERICAL_FEATURES),
        ('cat', categorical_transformer, config.CATEGORICAL_FEATURES)
    ]
)

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
titanic_pipe = Pipeline(
    steps = [
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ]
 )
