from airflow import DAG
from sqlalchemy_utils import database_exists, create_database
from airflow.operators.python_operator import PythonOperator

import re
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text


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
            'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide', 'metformin-pioglitazone',
            'metformin-rosiglitazone', 'glimepiride-pioglitazone', 'glipizide-metformin',
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
        'readmitted': {'>30': 0, '<30': 1, 'NO': 0}
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

    return df


def harmonize_dataframes(df_train, df_test, df_validation):
    # Unir las tres columnas de los dataframes
    cols = set(df_train.columns).union(df_test.columns).union(df_validation.columns)

    # Para cada dataframe, agregar las columnas que faltan y llenarlas con 0
    for df in [df_train, df_test, df_validation]:
        missing_cols = cols - set(df.columns)
        for col in missing_cols:
            df[col] = 0

    return df_train, df_test, df_validation

def load_clean_data(df_train, df_test, df_validation):
    POSTGRES_USER= "postgres"
    POSTGRES_PASSWORD="postgres"
    POSTGRES_DB= "clean_diabetes_db"
    POSTGRES_SERVER="clean-data"

    connection_string= "postgresql+psycopg2://"+ POSTGRES_USER + ":" + POSTGRES_PASSWORD + "@" + POSTGRES_SERVER + "/" + POSTGRES_DB
    engine = create_engine(connection_string)

    # Create the database if it does not exist
    if not database_exists(engine.url):
        create_database(engine.url)
        print("[INFO] No existe la base de datos")
        # Escribe los datos de prueba y validación en la base de datos}
        df_train.to_sql('train_data', engine, if_exists='replace', index = False)
        df_test.to_sql('test_data', engine, if_exists='replace', index = False)
        df_validation .to_sql('validation_data', engine, if_exists='replace', index = False)
    else:
        # Escribe los datos de prueba y validación en la base de datos}
        df_train.to_sql('train_data', engine, if_exists='replace', index = False)
        df_test.to_sql('test_data', engine, if_exists='replace', index = False)
        df_validation .to_sql('validation_data', engine, if_exists='replace', index = False)

def clean_data():

    POSTGRES_USER= "postgres"
    POSTGRES_PASSWORD="postgres"
    POSTGRES_DB= "raw_diabetes_db"
    POSTGRES_SERVER="raw-data"

    connection_string= "postgresql+psycopg2://"+ POSTGRES_USER + ":" + POSTGRES_PASSWORD + "@" + POSTGRES_SERVER + "/" + POSTGRES_DB
    engine = create_engine(connection_string)

    query_train = 'SELECT * FROM train_data'
    query_test = 'SELECT * FROM test_data'
    query_validation = 'SELECT * FROM validation_data'

    with engine.connect() as conn:
        df_train = pd.read_sql_query(sql=text(query_train), con=conn)
        df_test = pd.read_sql_query(sql=text(query_test), con=conn)
        df_validation = pd.read_sql_query(sql=text(query_validation), con=conn)
    
    df_train = proccess_data(df_train)
    df_test = proccess_data(df_test)
    df_validation = proccess_data(df_validation)

    df_train, df_test, df_validation = harmonize_dataframes(df_train, df_test, df_validation)

    load_clean_data(df_train, df_test, df_validation)

# Definir el DAG
dag = DAG('2_clean_database',
        description="Procesar información datos diabetes",
        start_date=datetime(2023, 5, 28))

# Definir una tarea en el DAG
process_task = PythonOperator(task_id='clean_data', python_callable=clean_data, dag=dag)