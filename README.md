# Machine Learning Challenge

### The Challenge
Based on the well known Kagle problem "Titanic: Machine Learning from Disaster", develop two process:
- Train a classifier
- Create an API

#### Train a classifier
Create a pipeline for training a binary classifier. The input for this training can be found in the
link from above. The output should be a binary classifier model and exported somewhere so it
can be used from the API.

#### Create an API
After exporting the classifier as a binary, it is required to use it for real time predictions. With the
mentioned objective in mind, you would need to create an API that receives a list of passengers
and predicts for each of them if they’re likely to survive or not.

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
