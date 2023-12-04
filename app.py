# 1. Library imports
import uvicorn
from fastapi import FastAPI
from New_customer import new_customer
import pickle
import pandas as pd

# Version de FastAPI

app = FastAPI()

pickle_in = open("classifier_lr_few.pkl","rb")
classifier_pipeline = pickle.load(pickle_in)

data = pd.read_csv("data_loan_few_features.csv")
stored_data = {}

# 1. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Bienvenue sur mon API de prediction'}

# 2. Expose the prediction functionality, make a prediction from the passed
@app.post('/predict_proba/{client_id}')
def predict_client(client_id : int):
    row = data.loc[data['SK_ID_CURR'] == client_id]
    if row.empty:
        return { "message" : 'Customer do not exists' }
    prediction = classifier_pipeline.predict_proba(row)
    row.fillna("null", inplace=True)

    if prediction[0][0] > 0.67:
        prediction_text = "Great"
    else:
        prediction_text = "Need further evaluation"

    return {
        'prediction': prediction_text,
        'probabilité': prediction[0][0],
        'dataframe': row.to_dict()
    }

@app.post("/predict_new")
def predict_newcustomer(data:new_customer):
    data = data.dict()
    EXT_SOURCE_3= data['EXT_SOURCE_3']
    EXT_SOURCE_2= data['EXT_SOURCE_2']
    EXT_SOURCE_1= data['EXT_SOURCE_1']
    AMT_GOODS_PRICE= data['AMT_GOODS_PRICE']
    AMT_ANNUITY= data['AMT_ANNUITY']
    FLAG_OWN_CAR= data['FLAG_OWN_CAR']
    NAME_EDUCATION_TYPE= data['NAME_EDUCATION_TYPE']
    AMT_CREDIT= data['AMT_CREDIT']
    DAYS_EMPLOYED= data['DAYS_EMPLOYED']
    DAYS_BIRTH = data['DAYS_BIRTH']

    CREDIT_TERM = AMT_ANNUITY / AMT_CREDIT

    input_data_df = pd.DataFrame({
        'EXT_SOURCE_3': [EXT_SOURCE_3],
        'EXT_SOURCE_2': [EXT_SOURCE_2],
        'EXT_SOURCE_1': [EXT_SOURCE_1],
        'AMT_GOODS_PRICE': [AMT_GOODS_PRICE],
        'AMT_ANNUITY': [AMT_ANNUITY],
        'FLAG_OWN_CAR': [FLAG_OWN_CAR],
        'NAME_EDUCATION_TYPE': [NAME_EDUCATION_TYPE],
        'CREDIT_TERM': [CREDIT_TERM],
        'DAYS_EMPLOYED': [DAYS_EMPLOYED],
        'DAYS_BIRTH': [DAYS_BIRTH]
    })

    prediction = classifier_pipeline.predict_proba(input_data_df)

    dataframe_dict = {
        'EXT_SOURCE_3': EXT_SOURCE_3,
        'EXT_SOURCE_2': EXT_SOURCE_2,
        'CREDIT_TERM': CREDIT_TERM,
        'EXT_SOURCE_1': EXT_SOURCE_1,
        'AMT_GOODS_PRICE': AMT_GOODS_PRICE,
        'AMT_CREDIT' : AMT_CREDIT,
        'AMT_ANNUITY': AMT_ANNUITY,
        'FLAG_OWN_CAR': FLAG_OWN_CAR,
        'NAME_EDUCATION_TYPE': NAME_EDUCATION_TYPE,
        'DAYS_EMPLOYED': DAYS_EMPLOYED,
        'DAYS_BIRTH': DAYS_BIRTH

    }

    stored_data['dataframe_dict'] = dataframe_dict

    return {
        'probabilité': prediction[0][0],
        'dataframe_dict': dataframe_dict
    }

@app.get("/predict_new")
def get_stored_data():
    return stored_data

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run("app:app", host='127.0.0.1', port=8000)

#uvicorn app:app --reload
