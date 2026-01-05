import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("FEATURES")

def load_raw_data(path):
    logger.info("Loading processed dataset")
    return pd.read_csv(path)

if __name__ == "__main__":
    df = load_raw_data("data/raw/Messages.csv")
    print(df.head())
