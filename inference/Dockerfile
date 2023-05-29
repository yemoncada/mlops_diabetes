FROM python:3.9-slim

WORKDIR /mlflow/

RUN apt-get update \
    && apt-get -y install libpq-dev gcc python3-dev\
    && pip install psycopg2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
  rm requirements.txt

EXPOSE 5000

ENV BACKEND_URI postgresql+psycopg2://postgres:postgres@localhost/mlflow_db
ENV ARTIFACT_ROOT s3://mlflow3/artifacts
ENV AWS_ACCESS_KEY_ID your_key
ENV AWS_ACCESS_KEY_ID your_key

CMD mlflow server --backend-store-uri ${BACKEND_URI} --default-artifact-root ${ARTIFACT_ROOT} --host 0.0.0.0 --port 5000 --serve-artifacts