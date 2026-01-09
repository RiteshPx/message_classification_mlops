FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN dvc pull models/best_model.pkl
RUN dvc pull artifacts/vectorizer/tfidf.pkl

COPY src/ src/
COPY app.py .
COPY models/ models/
COPY artifacts/ artifacts/

EXPOSE 7000
CMD ["python", "app.py"]

 