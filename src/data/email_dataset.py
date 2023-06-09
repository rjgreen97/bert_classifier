from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd


class EmailDataset(Dataset):
    TOKENIZER_NAME = "bert-base-uncased"

    def __init__(self, csv_path, max_len=512):
        self.df = self._csv_to_df(csv_path)
        self.tokenizer = BertTokenizer.from_pretrained(
            self.TOKENIZER_NAME, do_lower_case=True
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

    def _csv_to_df(self, csv_path) -> pd.DataFrame:
        df = pd.read_csv(csv_path, encoding="utf-8")
        df_proccessed = df[["label", "text"]]
        return df_proccessed

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
