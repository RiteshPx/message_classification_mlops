import yaml
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.utils.logger import setup_logger

logger = setup_logger("FEATURES")


def load_config():
    with open("src/config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def build_features():
    config = load_config()

    df = pd.read_csv(config["data"]["processed"])
    logger.info("Loading processed dataset")

    X = df["Text"]
    y = df["Label"]

    logger.info("Splitting data into train/test")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"]
    )

    logger.info("Fitting TF-IDF on training data only")
    logger.info("Transforming test data")
    vectorizer = TfidfVectorizer(
        max_features=config["features"]["max_features"],
        ngram_range=tuple(config["features"]["ngram_range"])
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Save artifacts
    logger.info("Saving vectorizer and feature artifacts")
    joblib.dump(vectorizer, config["features"]["vectorizer_path"])
    joblib.dump((X_train, y_train), config["features"]["train_features"])
    joblib.dump((X_test, y_test), config["features"]["test_features"])


if __name__ == "__main__":
    build_features()
