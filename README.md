# Rappi - Machine Learning Challenge

### Dependencies
```
pip install -r packages/requirements.txt
```

### Train Pipeline
```
export PYTHONPATH=$PYTHONPATH:$PWD/packages/api:$PWD/packages/titanic
cd packages/titanic
python titanic/train_pipeline.py
```

### Running API
```
export PYTHONPATH=$PYTHONPATH:$PWD/packages/api:$PWD/packages/titanic
cd packages/api
gunicorn -b :<port> run:application
```
### Build Docker Image
```
docker build -t <image_name> .
```

### Run Docker Image
```
docker run -p <port>:5000 <image_name>
```

# Endpoints

### /healt (GET)
Basic sanity check.

### /predict (POST)
Receives a list of passengers and predicts for each of them if they’re likely to survive or not. It also return the binary classifier used for the prediction. Here's a sample input:
```
[
    {"Age": 85, "Sex": "female", "Embarked": "C", "Pclass": 3},
    {"Age": 24, "Sex": "female", "Embarked": "C", "Pclass": 2},
    {"Age": 3,  "Sex": "male",   "Embarked": "C", "Pclass": 1},
    {"Age": 44, "Sex": "female", "Embarked": "Q", "Pclass": 3}
]
```

and sample output:
```
{
  "model": "Logistic",
  "predictions": [
    0,
    1,
    1,
    1
  ]
}
```
