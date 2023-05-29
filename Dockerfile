FROM apache/airflow:2.6.1-python3.9

ADD requirements.txt . 

RUN python -m pip install --upgrade pip
RUN python -m pip install psycopg2-binary

RUN pip install -r requirements.txt 
