import yaml
import joblib
import numpy as np
from flask import Flask, request, jsonify,render_template

from src.utils.logger import setup_logger

# --------------------------------------------------
# Setup
# --------------------------------------------------
app = Flask(__name__)
logger = setup_logger("FLASK")

# --------------------------------------------------
# Load config
# --------------------------------------------------
with open("src/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["models"]["best_model_path"]
VECTORIZER_PATH = config["features"]["vectorizer_path"]

# --------------------------------------------------
# Load artifacts ONCE (important)
# --------------------------------------------------
logger.info("Loading model and vectorizer")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

logger.info("Model and vectorizer loaded successfully")

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/')
def home():
    return render_template('index.html')

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # data1 = request.get_json()  # use in case of JSON input
        data = request.form.get("text")
        # logger.info(f"Input data: {data}")

        if not data :
            logger.warning("Invalid input received")
            return jsonify({"error": "Missing 'text' field"}), 400

        # text = data["text"]
        text = data
        logger.info("Received prediction request")

        # Vectorize input
        X = vectorizer.transform([text])

        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0].max()

        label = "Spam" if prediction == 1 else "Ham"

        # return jsonify({
        #     "prediction": label,
        #     "confidence": round(float(probability), 4)
        # })
        return render_template('result.html', 
                           prediction=label, 
                           message=text,
                           confidence=round(float(probability), 2)*100)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=False)
    # app.run(debug=True, port=7000)  # For local testing

