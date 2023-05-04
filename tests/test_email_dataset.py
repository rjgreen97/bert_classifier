# import pytest
from transformers import BertTokenizer

from src.data.email_dataset import EmailDataset
from utils.data_prep import process_csv

df = process_csv("tests/fixtures/emails.csv")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
email_dataset = EmailDataset(df, tokenizer, max_len=8)


def test_init():
    assert email_dataset.df.equals(df)
    assert email_dataset.tokenizer == tokenizer
    assert email_dataset.max_len == 8


def test_len():
    assert len(email_dataset) == 10


def test_getitem():
    assert email_dataset[1]["input_ids"].tolist() == [
        101,
        2023,
        2003,
        1037,
        12403,
        2213,
        10373,
        102,
    ]
    assert email_dataset[1]["attention_mask"].tolist() == [1, 1, 1, 1, 1, 1, 1, 1]
    assert email_dataset[1]["label"] == 1


def test_tokenize():
    tokenized_input = email_dataset._tokenize(email_dataset.df.text[0])
    input_ids = tokenized_input["input_ids"].flatten()
    attention_mask = tokenized_input["attention_mask"].flatten()
    assert input_ids.tolist() == [101, 2023, 2003, 1037, 2613, 10373, 102, 0]
    assert attention_mask.tolist() == [1, 1, 1, 1, 1, 1, 1, 0]


def test_parse_labels():
    assert email_dataset._parse_labels("spam") == 1
    assert email_dataset._parse_labels("ham") == 0
    assert email_dataset[0]["label"] == 0


if __name__ == "__main__":
    test_init()
    test_len()
    test_getitem()
    test_tokenize()
    test_parse_labels()
