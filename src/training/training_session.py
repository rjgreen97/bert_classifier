import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer

from src.data.email_dataset import EmailDataset
from src.data.email_dataset_splitter import EmailDatasetSplitter
from src.training.trainer import Trainer
from src.training.training_session_arg_parser import TrainingSessionArgParser
from src.utils.data_preprocessing import process_csv


class TrainingSession:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        self.create_datasets()
        self.create_dataloaders()
        self.create_model()
        self.create_optimizer()
        self.create_trainer()
        self.trainer.run()

    def create_datasets(self):
        df = process_csv(self.args.data_path)
        tokenizer = BertTokenizer.from_pretrained(
            self.args.tokenizer, do_lower_case=True
        )
        dataset_splitter = EmailDatasetSplitter(EmailDataset(df, tokenizer))
        self.train_dataset, self.val_dataset = dataset_splitter.random_split()

    def create_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size
        )

    def create_model(self):
        self.model = (
            BertForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self.args.model_name,
                num_labels=self.args.output_classes,
                output_attentions=False,
                output_hidden_states=False,
            )
        ).to(self.device)

    def create_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def create_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            optimizer=self.optimizer,
        )


if __name__ == "__main__":
    args = TrainingSessionArgParser().parse_args()
    session = TrainingSession(args)
    session.run()