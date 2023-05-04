import torch
from torch.nn import BCELoss


class Trainer:
    def __init__(
        self, model, train_dataloader, val_dataloader, batch_size, epochs, optimizer
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = BCELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        print(f"Training started on {self.device}.")
        for _epoch in range(self.epochs):
            self.train()
            self.validate()

    def train(self):
        pass

    def validate(self):
        pass
