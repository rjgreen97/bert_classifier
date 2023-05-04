from src.data.email_dataset import EmailDataset
from src.utils.data_prep import process_csv
from transformers import BertTokenizer

if __name__ == "__main__":
    df = process_csv("data/raw_emails.csv")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    email_dataset = EmailDataset(df, tokenizer)
