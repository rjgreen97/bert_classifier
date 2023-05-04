import pytest
from torch.utils.data import random_split
from transformers import BertTokenizer

from src.data.email_dataset import EmailDataset
from src.data.email_dataset_splitter import EmailDatasetSplitter
from utils.data_prep import process_csv

df = process_csv("tests/fixtures/emails.csv")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
email_dataset = EmailDataset(df, tokenizer, max_len=8)


def test_init():
    dataset_splitter = EmailDatasetSplitter(email_dataset, val_size=0.5)
    assert dataset_splitter.dataset == email_dataset
    assert dataset_splitter.val_size == 0.5


def test_random_split_with_valid_val_size():
    dataset_splitter = EmailDatasetSplitter(email_dataset, val_size=0.5)
    train_dataset, val_dataset = dataset_splitter.random_split()
    assert len(train_dataset) == 5
    assert len(val_dataset) == 5


def test_random_split_with_invalid_val_size():
    dataset_splitter = EmailDatasetSplitter(email_dataset, val_size=1.5)
    with pytest.raises(ValueError):
        dataset_splitter.random_split()


if __name__ == "__main__":
    test_init()
    test_random_split_with_valid_val_size()
    test_random_split_with_invalid_val_size()
