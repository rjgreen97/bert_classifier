from argparse import ArgumentParser


class TrainingSessionArgParser:
    def __init__(self):
        self.parser = ArgumentParser(description="Training session arguments")
        self.add_args()

    def add_args(self):
        self.parser.add_argument(
            "--data_path",
            type=str,
            default="data/raw_emails.csv",
            help="Filepath to the raw csv data.",
        )
        self.parser.add_argument(
            "--model_checkpoint_path",
            type=str,
            default="./model.pth",
            help="Filepath to saved model checkpoints.",
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=12,
            help="Batch size to be used during training.",
        )
        self.parser.add_argument(
            "--epochs",
            type=int,
            default=3,
            help="Number of full training passes over the entire dataset.",
        )
        self.parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-4,
            help="Learning rate to be used during training.",
        )
        self.parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.01,
            help="Weight decay to be used in the optimizer.",
        )
        self.parser.add_argument(
            "--patience",
            type=int,
            default=5,
            help="Number of epochs without improvement after which training stops.",
        )

    def parse_args(self):
        return self.parser.parse_args()
