FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5001
ENV PREDICTOR_API=http://prediction-api:5000/predict
CMD ["python", "app.py"]
