import re
import pandas as pd
from src.utils.logger import setup_logger
logger = setup_logger("PREPROCESS")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_data(input_path, output_path):
    logger.info("Starting preprocessing stage")
    logger.info(f"Reading raw data from {input_path}")
    df = pd.read_csv(input_path)

    logger.info(f"Raw data shape: {df.shape}")
    df["Text"] = df["Message"].astype(str).apply(clean_text)

    df["Label"] = df["Label"].map({"ham": 0, "spam": 1})
    df=df[["Label","Text"]]

    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    preprocess_data(
        "data/raw/Messages.csv",
        "data/preprocessed/cleaned.csv"
    )
