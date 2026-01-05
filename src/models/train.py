import yaml
import joblib
import os
import json

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from src.utils.logger import setup_logger

logger = setup_logger("TRAIN")

def load_config():
    with open("src/config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds)
    }


def main():
    config = load_config()

    logger.info("Loading training and test features")
    X_train, y_train = joblib.load("artifacts/X_train.pkl")
    X_test, y_test = joblib.load("artifacts/X_test.pkl")

    results = {}
    best_score = -1
    best_model = None

    os.makedirs(config["models"]["output_dir"], exist_ok=True)

    # -------- Naive Bayes --------
    logger.info("Starting model training and hyperparameter tuning")
    logger.info("Training Naive Bayes")
    for alpha in config["models"]["candidates"]["naive_bayes"]["params"]["alpha"]:
        logger.info(f"Training Naive Bayes with alpha={alpha}")
        model = MultinomialNB(alpha=alpha)
        metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test)
        logger.info(f"Metrics: {metrics}")  

        results[f"nb_alpha_{alpha}"] = metrics

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_model = model

    # -------- svm --------
    logger.info("Training SVM")
    for C in config["models"]["candidates"]["svc"]["params"]["C"]:
        model = SVC(C=C)
        logger.info(f"Training Logistic Regression with C={C}")
        metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test)
            
        results[f"lr_C_{C}"] = metrics
        logger.info(f"Metrics: {metrics}")


        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_model = model

    #-------------random forest ---------
    logger.info("Training Random Forest")
    for n_estimators in config["models"]["candidates"]["random_forest"]["params"]["n_estimators"]:
        model = RandomForestClassifier(n_estimators=n_estimators)
        metrics = train_and_evaluate(model, X_train, y_train, X_test, y_test)

        results[f"rf_n_estimators_{n_estimators}"] = metrics
        logger.info(f"Metrics: {metrics}")

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_model = model

    # Save best model
    logger.info("Selecting best model based on F1 score")
    logger.info("Saving best model artifact")
    joblib.dump(best_model, config["models"]["best_model_path"])

    # Save metrics
    # joblib.dump(results, "artifacts/metrics/model_metrics.pkl")
    with open(config["metrics"]["path"], "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Training stage completed successfully")
    


if __name__ == "__main__":
    main()
