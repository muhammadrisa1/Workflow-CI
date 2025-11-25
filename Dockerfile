FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install mlflow scikit-learn pandas numpy

# Install MLServer for serving
RUN pip install mlserver mlserver-sklearn

# Default command
CMD ["python", "-c", "print('MLflow model container ready')"]
