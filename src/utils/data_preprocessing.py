import pandas as pd


def process_csv(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="utf-8")
    df_proccessed = df[["label", "text"]]
    return df_proccessed


if __name__ == "__main__":
    df = process_csv("data/raw_emails.csv")
