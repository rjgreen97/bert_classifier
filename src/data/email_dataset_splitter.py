from torch.utils.data import random_split
from transformers import BertTokenizer

from src.data.email_dataset import EmailDataset
from utils.data_prep import process_csv


class EmailDatasetSplitter:
    def __init__(self, dataset, val_size=0.2):
        self.dataset = dataset
        self.val_size = val_size

    def random_split(self) -> tuple:
        if self.val_size < 0 or self.val_size > 1:
            raise ValueError(f"val_size '{self.val_size}' must be between 0 and 1")
        self.val_size = int(len(self.dataset) * self.val_size)
        train_size = len(self.dataset) - self.val_size
        return random_split(self.dataset, [train_size, self.val_size])


if __name__ == "__main__":
    df = process_csv("data/raw_emails.csv")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    dataset_splitter = EmailDatasetSplitter(EmailDataset(df, tokenizer))
    train_dataset, val_dataset = dataset_splitter.random_split()
