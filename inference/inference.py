from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import numpy as np
import re
import os
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sqlalchemy import create_engine
import psycopg2
from psycopg2 import sql

app = FastAPI()

os.environ['AWS_ACCESS_KEY_ID'] = 'AWS_SECRET_ACCESS_KEY'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'AWS_SECRET_ACCESS_KEY'

def proccess_data(df):

    drop_Idx = set(df[(df['diag_1'] == '?') & (df['diag_2'] == '?') & (df['diag_3'] == '?')].index)

    drop_Idx = drop_Idx.union(set(df['diag_1'][df['diag_1'] == '?'].index))
    drop_Idx = drop_Idx.union(set(df['diag_2'][df['diag_2'] == '?'].index))
    drop_Idx = drop_Idx.union(set(df['diag_3'][df['diag_3'] == '?'].index))
    drop_Idx = drop_Idx.union(set(df['race'][df['race'] == '?'].index))
    drop_Idx = drop_Idx.union(set(df[df['discharge_disposition_id'] == 11].index))
    drop_Idx = drop_Idx.union(set(df['gender'][df['gender'] == 'Unknown/Invalid'].index))
    new_Idx = list(set(df.index) - set(drop_Idx))
    df = df.iloc[new_Idx]

    # Create new temporary columns and compute numchange
    keys = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose',
            'miglitol', 'insulin', 'glyburide_metformin', 'tolazamide', 'metformin_pioglitazone',
            'metformin_rosiglitazone', 'glimepiride_pioglitazone', 'glipizide_metformin',
            'troglitazone', 'tolbutamide', 'acetohexamide']

    df['numchange'] = sum((df[col].map({'No': 0, 'Steady': 0}).fillna(1) for col in keys))

    # Replace id mappings in specific columns
    id_mappings = {
        'admission_type_id': {2:1, 7:1, 6:5, 8:5},
        'discharge_disposition_id': {6:1, 8:1, 9:1, 13:1, 3:2, 4:2, 5:2, 14:2, 22:2, 23:2, 24:2, 12:10, 15:10, 16:10, 17:10, 25:18, 26:18},
        'admission_source_id': {2:1, 3:1, 5:4, 6:4, 10:4, 22:4, 25:4, 15:9, 17:9, 20:9, 21:9, 13:11, 14:11}
    }

    for col, mapping in id_mappings.items():
        df[col] = df[col].replace(mapping)

    # Replace specific string values in columns
    string_mappings = {
        'change': {'Ch': 1, 'No': 0},
        'gender': {'Male': 1, 'Female': 0},
        'diabetesMed': {'Yes': 1, 'No': 0},
        'A1Cresult': {'>7': 1, '>8': 1, 'Norm': 0, 'None': -99},
        'max_glu_serum': {'>200': 1, '>300': 1, 'Norm': 0, 'None': -99},
    }

    for col, mapping in string_mappings.items():
        df[col] = df[col].replace(mapping)

    # Replace medicine related columns with new mappings
    medicine_mappings = {'No': 0, 'Steady': 1, 'Up': 1, 'Down': 1}
    df[keys] = df[keys].replace(medicine_mappings)

    # age intervals [0-10) - [90-100) from 1-10
    df['age'] = df['age'].replace({f'[{10*i}-{10*(i+1)})': i+1 for i in range(10)})

    # log transformation on specific columns
    log_columns = ['number_outpatient', 'number_inpatient', 'number_emergency']
    df[log_columns] = np.log1p(df[log_columns])

    # Fill NaN values in diag_list
    diag_list = ['diag_1','diag_2','diag_3']
    df[diag_list] = df[diag_list].fillna('NaN')

    # Dropping columns with large number of missing values
    drop_list = ['citoglipton', 'examide', 'weight','encounter_id', 'patient_nbr','payer_code','medical_specialty']
    df = df.drop(drop_list, axis = 1)

    # Refactoring diagnosis categories
    def transform_diag(value):
        value = re.sub("V[0-9]*", "0", value) 
        value = re.sub("E[0-9]*", "0", value) 
        value = re.sub('NaN', "-1", value) 
        value = float(value)

        if 390 <= value <= 459 or value == 785:
            return 'Circulatory'
        elif 460 <= value <= 519 or value == 786:
            return 'Respiratory'
        elif 520 <= value <= 579 or value == 787:
            return 'Digestive'
        elif value == 250:
            return 'Diabetes'
        elif 800 <= value <= 999:
            return 'Injury'
        elif 710 <= value <= 739:
            return 'Musculoskeletal'
        elif 580 <= value <= 629 or value == 788:
            return 'Genitourinary'
        elif 140 <= value <= 239:
            return 'Neoplasms'
        elif value == -1:
            return 'NAN'
        else:
            return 'Other'

    df[diag_list] = df[diag_list].applymap(transform_diag)

    # One hot Encoding Race and Id's 
    df = pd.get_dummies(df, columns=['race'], prefix=["enc"])

    columns_ids = ['diag_1','diag_2','diag_3','admission_type_id', 'discharge_disposition_id', 'admission_source_id']

    df[columns_ids] = df[columns_ids].astype('str')
    df = pd.get_dummies(df, columns=columns_ids)

    df.rename(columns={
        "glimepiride_pioglitazone": "glimepiride-pioglitazone",
        "glipizide_metformin": "glipizide-metformin",
        "glyburide_metformin": "glyburide-metformin",
        "metformin_pioglitazone": "metformin-pioglitazone",
        "metformin_rosiglitazone":"metformin-rosiglitazone"}, errors="raise", inplace = True)
    
    print(df.columns)
    
    train_columns = ['index','gender', 'age', 'time_in_hospital', 'num_lab_procedures',
       'num_procedures', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient', 'number_diagnoses',
       'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
       'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
       'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
       'miglitol', 'troglitazone', 'tolazamide', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'change', 'diabetesMed',
       'numchange', 'enc_AfricanAmerican', 'enc_Asian', 'enc_Caucasian',
       'enc_Hispanic', 'enc_Other', 'diag_1_Circulatory', 'diag_1_Diabetes',
       'diag_1_Digestive', 'diag_1_Genitourinary', 'diag_1_Injury',
       'diag_1_Musculoskeletal', 'diag_1_Neoplasms', 'diag_1_Other',
       'diag_1_Respiratory', 'diag_2_Circulatory', 'diag_2_Diabetes',
       'diag_2_Digestive', 'diag_2_Genitourinary', 'diag_2_Injury',
       'diag_2_Musculoskeletal', 'diag_2_Neoplasms', 'diag_2_Other',
       'diag_2_Respiratory', 'diag_3_Circulatory', 'diag_3_Diabetes',
       'diag_3_Digestive', 'diag_3_Genitourinary', 'diag_3_Injury',
       'diag_3_Musculoskeletal', 'diag_3_Neoplasms', 'diag_3_Other',
       'diag_3_Respiratory', 'admission_type_id_1', 'admission_type_id_3',
       'admission_type_id_4', 'admission_type_id_5',
       'discharge_disposition_id_1', 'discharge_disposition_id_10',
       'discharge_disposition_id_18', 'discharge_disposition_id_19',
       'discharge_disposition_id_2', 'discharge_disposition_id_20',
       'discharge_disposition_id_27', 'discharge_disposition_id_28',
       'discharge_disposition_id_7', 'admission_source_id_1',
       'admission_source_id_11', 'admission_source_id_4',
       'admission_source_id_7', 'admission_source_id_8',
       'admission_source_id_9']

    # Encuentra las columnas que están en df1 pero no en df2
    missing_cols = set(train_columns) - set(df.columns)

    # Añade estas columnas a df2, llenándolas con el valor 1
    for c in missing_cols:
        df[c] = 1

    # Asegúrate de que las columnas estén en el mismo orden que en df1
    df = df[train_columns]

    return df

class PatientData(BaseModel):
    encounter_id: int
    patient_nbr: int
    race: str
    gender: str
    age: str
    weight: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    payer_code: str
    medical_specialty: str
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    diag_1: str
    diag_2: str
    diag_3: str
    number_diagnoses: int
    max_glu_serum: str
    A1Cresult: str
    metformin: str
    repaglinide: str
    nateglinide: str
    chlorpropamide: str
    glimepiride: str
    acetohexamide: str
    glipizide: str
    glyburide: str
    tolbutamide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    miglitol: str
    troglitazone: str
    tolazamide: str
    examide: str
    citoglipton: str
    insulin: str
    glyburide_metformin: str
    glipizide_metformin: str
    glimepiride_pioglitazone: str
    metformin_rosiglitazone: str
    metformin_pioglitazone: str
    change: str
    diabetesMed: str

@app.on_event("startup")
def load_model():

    POSTGRES_USER= "postgres"
    POSTGRES_PASSWORD="postgres"
    POSTGRES_DB= "raw_diabetes_db"
    POSTGRES_SERVER="raw-data"

    global client
    # connects to the Mlflow tracking server that you started above
    mlflow.set_tracking_uri("http://mlflow:5000")
    # Configurar el cliente de MLflow
    client = MlflowClient()

    global conn
    connection_string= "postgresql+psycopg2://"+ POSTGRES_USER + ":" + POSTGRES_PASSWORD + "@" + POSTGRES_SERVER + "/" + POSTGRES_DB
    engine = create_engine(connection_string)
    conn = engine.connect()

@app.post("/predict")
def predict(data: PatientData):

    stage = "Production"
    model_names = []

    for rm in client.search_registered_models():
        model_name = rm.name
        model_names.append(model_name)

    if not len(model_names) == 0:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
        # Obtener la última versión del modelo en la etapa de producción
        model_version = client.get_latest_versions(model_name, stages=["Production"])[0]

        # Obtener el run_id y el run_name del modelo en producción
        run_id = model_version.run_id
        run = client.get_run(run_id)
        run_name = run.data.tags["mlflow.runName"]

        data = data.dict()
        df = pd.DataFrame(data, index=[0])  
        df  = proccess_data(df)
        # convierte los datos en un dataframe de pandas
        prediction = model.predict(df)  # realiza la predicción

        return {"prediction": int(prediction[0]),
                "model_version": run_name}
    else:
        raise HTTPException(status_code=500, detail='No existen modelos en producción cargados')
    


 