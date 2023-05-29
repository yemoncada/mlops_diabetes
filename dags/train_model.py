from datetime import datetime
import pandas as pd
import mlflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from sk_models import MODELS
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy import create_engine, text
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os


os.environ['AWS_ACCESS_KEY_ID'] = 'AWS_ACESS_KEY'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'AWS_ACESS_KEY'

def load_clean_data():
    POSTGRES_USER= "postgres"
    POSTGRES_PASSWORD="postgres"
    POSTGRES_DB= "clean_diabetes_db"
    POSTGRES_SERVER="clean-data"

    connection_string= "postgresql+psycopg2://"+ POSTGRES_USER + ":" + POSTGRES_PASSWORD + "@" + POSTGRES_SERVER + "/" + POSTGRES_DB
    engine = create_engine(connection_string)

    query_train = 'SELECT * FROM train_data'
    query_test = 'SELECT * FROM test_data'
    query_validation = 'SELECT * FROM validation_data'

    with engine.connect() as conn:
        df_clean_train = pd.read_sql_query(sql=text(query_train), con=conn)
        df_clean_test = pd.read_sql_query(sql=text(query_test), con=conn)
        df_clean_validation = pd.read_sql_query(sql=text(query_validation), con=conn)
    
    return df_clean_train, df_clean_test, df_clean_validation

def train_model():

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("mlflow_diabetes_exp")

    df_clean_train, df_clean_test, df_clean_validation = load_clean_data()

    X_train = df_clean_train.drop('readmitted', axis=1)
    y_train = df_clean_train.readmitted
    X_test = df_clean_test.drop('readmitted', axis=1)
    y_test = df_clean_test.readmitted

        # Itera sobre tus modelos y hiperparámetros
    for model_info in MODELS:
        with mlflow.start_run(run_name=model_info['name']):
            for iter, params in enumerate(model_info['params']):
                with mlflow.start_run(run_name= model_info['name'] + '_' + 'it' + str(iter), nested=True) as nested:
                    # Entrenamiento del modelo
                    model = model_info['model'](**params)
                    model.fit(X_train, y_train)

                    # Predicción y evaluación del modelo
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')

                    # Registra los parámetros, métricas y modelo en MLflow
                    mlflow.log_params(params)
                    mlflow.log_metrics({'accuracy': accuracy, 'precision': precision, 'recall': recall})
                    mlflow.sklearn.log_model(model, "model")

# Definir el DAG
dag = DAG('3_train_model',
        description="Entrenar el modelo y guardar en mlflow",
        start_date=datetime(2023, 5, 28))

# Definir una tarea en el DAG
train_task = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)

