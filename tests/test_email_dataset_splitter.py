import pytest

from src.data.email_dataset import EmailDataset
from src.data.email_dataset_splitter import EmailDatasetSplitter

email_dataset = EmailDataset("tests/fixtures/emails.csv", max_len=8)


def test_init():
    dataset_splitter = EmailDatasetSplitter(email_dataset)
    assert dataset_splitter.dataset == email_dataset


def test_random_split_with_valid_val_size():
    dataset_splitter = EmailDatasetSplitter(email_dataset)
    train_dataset, val_dataset = dataset_splitter.random_split(val_size=0.5)
    assert len(train_dataset) == 5
    assert len(val_dataset) == 5


def test_random_split_with_invalid_val_size():
    dataset_splitter = EmailDatasetSplitter(email_dataset)
    with pytest.raises(ValueError):
        dataset_splitter.random_split(val_size=1.5)
