FROM python:3.8-slim

WORKDIR /app

COPY MLProject/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY MLProject/ .

RUN pip install mlserver-sklearn

RUN echo '@@servers: [{"name": "sklearn", "implementation": "mlserver_sklearn.SKLearnModel"}]' > model-settings.yml

CMD ["python", "modelling.py"]