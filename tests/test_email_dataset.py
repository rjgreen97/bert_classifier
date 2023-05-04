from transformers import BertTokenizer

from src.data.email_dataset import EmailDataset

csv_path = "tests/fixtures/emails.csv"


def test_init():
    dataset = EmailDataset(csv_path, max_len=8)
    assert dataset.df.shape == (10, 2)
    assert isinstance(dataset.tokenizer, BertTokenizer)
    assert dataset.max_len == 8


def test_len():
    dataset = EmailDataset(csv_path, max_len=8)
    assert len(dataset) == 10


def test_getitem():
    dataset = EmailDataset(csv_path, max_len=8)
    example = dataset[1]
    assert example["input_ids"].tolist() == [
        101,
        2023,
        2003,
        1037,
        12403,
        2213,
        10373,
        102,
    ]
    assert example["attention_mask"].tolist() == [1, 1, 1, 1, 1, 1, 1, 1]
    assert example["label"] == 1


def test_tokenize():
    dataset = EmailDataset(csv_path, max_len=8)
    tokenized_input = dataset._tokenize(dataset.df.text[0])
    input_ids = tokenized_input["input_ids"].flatten()
    attention_mask = tokenized_input["attention_mask"].flatten()
    assert input_ids.tolist() == [101, 2023, 2003, 1037, 2613, 10373, 102, 0]
    assert attention_mask.tolist() == [1, 1, 1, 1, 1, 1, 1, 0]


def test_parse_labels():
    dataset = EmailDataset(csv_path, max_len=8)
    assert dataset._parse_labels("spam") == 1
    assert dataset._parse_labels("ham") == 0
    assert dataset[0]["label"] == 0
