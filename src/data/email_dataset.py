from torch.utils.data import Dataset
from transformers import BertTokenizer

from src.utils.data_prep import process_csv


class EmailDataset(Dataset):
    def __init__(self, csv_path, max_len=512):
        self.df = process_csv(csv_path)
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> dict:
        email = self.df.text[index]
        label = self.df.label[index]
        tokenized_email = self._tokenize(email)
        parsed_label = self._parse_labels(label)

        item = {
            "input_ids": tokenized_email["input_ids"].flatten(),
            "attention_mask": tokenized_email["attention_mask"].flatten(),
            "label": parsed_label,
        }
        return item

    def _tokenize(self, email) -> dict:
        return self.tokenizer.encode_plus(
            email,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def _parse_labels(self, label) -> int:
        return 1 if label == "spam" else 0
