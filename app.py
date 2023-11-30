# -*- coding: utf-8 -*-

# 1. Library imports
import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd
import random
import streamlit as st

# 2. Create the app object
app = FastAPI()

try:
    with open("/Users/jeaneudesdesgraviers/Downloads/project 7/classifier_pipe_project_7.pkl", "rb") as pickle_file:
        classifier = pickle.load(pickle_file)
except FileNotFoundError:
    print("There is no pickle file to be loaded.")

try:
    with open("/Users/jeaneudesdesgraviers/Downloads/project 7/preprocessed_dataset.csv", "rb") as data_file:
        data = pd.read_csv(data_file)
except FileNotFoundError:
    print("There is no data file to be loaded.")

random_pick = random.randint(1,len(data))
random_row = data.iloc[random_pick]

st.title('Modèle de scoring')

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}

# 5. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_fraud():
    # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict(random_row)
    if (prediction[0] > 0.5):
        prediction = "Fake note"
    else:
        prediction = "Its a Bank note"
    return {
        'prediction': prediction
    }


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn app:app --reload