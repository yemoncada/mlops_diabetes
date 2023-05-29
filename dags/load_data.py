from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sqlalchemy import create_engine
from datetime import datetime
import requests
from io import BytesIO
import pandas as pd
from sqlalchemy_utils import database_exists, create_database
from sklearn.model_selection import train_test_split

def split_data(data, test_size=0.2, validation_size=0.2):
    train_data, test_data = train_test_split(data, test_size=test_size)
    train_data, validation_data = train_test_split(train_data, test_size=validation_size)
    return train_data, validation_data, test_data


def download_data():

    # URL del dataset
    url = 'https://docs.google.com/uc?export= \
    download&confirm={{VALUE}}&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC'

    # Hacer la petición
    r = requests.get(url)

    # Crea un objeto BytesIO a partir del contenido de la respuesta
    data = BytesIO(r.content)

    # Lee los datos en un DataFrame de pandas
    df = pd.read_csv(data)

    return df


def load_data_to_database():

    POSTGRES_USER= "postgres"
    POSTGRES_PASSWORD="postgres"
    POSTGRES_DB= "raw_diabetes_db"
    POSTGRES_SERVER="raw-data"

    connection_string= "postgresql+psycopg2://"+ POSTGRES_USER + ":" + POSTGRES_PASSWORD + "@" + POSTGRES_SERVER + "/" + POSTGRES_DB
    engine = create_engine(connection_string)

    # Create the database if it does not exist
    if not database_exists(engine.url):
        create_database(engine.url)

        print("[INFO] No existe la base de datos")
        
        df = download_data()
        print('Descargue la Data')

        # Divide los datos
        train_data, validation_data, test_data = split_data(df)

        print('Cargue la base de validación')

        # Escribe los datos de prueba y validación en la base de datos
        validation_data.to_sql('validation_data', engine, if_exists='replace', index = False)
        test_data.to_sql('test_data', engine, if_exists='replace', index = False)

        # Escribe los datos de entrenamiento en la base de datos por lotes
        batch_size = 15000
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            if i == 0:
                batch.to_sql('train_data', engine, if_exists='replace', index = False)
            else:
                batch.to_sql('train_data', engine, if_exists='append', index = False)
    
    else:
        print('[INFO] Base datos y contiene los datos cargados')

# Definir el DAG
dag = DAG('1_load_data',
          description="Cargue a la base de datos con la separacion train, test y validation",
          start_date=datetime(2023, 5, 28))

# Definir una tarea en el DAG
load_data_task = PythonOperator(task_id='load_data', python_callable=load_data_to_database, dag=dag)