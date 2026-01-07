import yaml
import joblib
import numpy as np
from flask import Flask, request, jsonify

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
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            logger.warning("Invalid input received")
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"]

        logger.info("Received prediction request")

        # Vectorize input
        X = vectorizer.transform([text])

        # Predict
        prediction = model.predict(X)[0]
        # probability = model.predict_proba(X)[0].max()

        label = "spam" if prediction == 1 else "ham"

        return jsonify({
            "prediction": label,
            # "confidence": round(float(probability), 4)
        })

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=False)
    # app.run(debug=True, port=7000)  # For local testing

