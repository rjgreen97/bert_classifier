import torch
from torch.nn import BCELoss
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        batch_size,
        epochs,
        optimizer,
        patience,
        model_checkpoint_path,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.patience = patience
        self.model_checkpoint_path = model_checkpoint_path
        self.criterion = BCELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        print(f"Training started on {self.device}.")
        for _epoch in range(self.epochs):
            self.train()
            self.validate()

    def train(self):
        self.model.train().to(self.device)
        best_validation_accuracy = 0

        for epoch in range(1, self.epochs):
            loop = tqdm(self.train_dataloader)
            for batch in loop:
                batch_input_ids = batch["input_ids"].to(self.device)
                batch_attention_mask = batch["attention_mask"].to(self.device)
                batch_label = batch["label"].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(
                    b_input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_label,
                    return_dict=True,
                )

                loss = self.criterion(outputs.logits, batch_label)
                loss.backward()
                self.optimizer.step()

                loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item())

            print(f"Training Loss: {loss.item():.3f}")

            accuracy = self.validate()
            if accuracy > best_validation_accuracy:
                epochs_no_improvement = 0
                torch.save(self.model.state_dict(), self.model_checkpoint_path)
                best_validation_accuracy = accuracy
            else:
                epochs_no_improvement += 1
                print(f"Epochs without improvement: {epochs_no_improvement}")
                if epochs_no_improvement > self.patience:
                    print(f"Early Stopping: Patience of {self.patience} reached.")
                    break

        print("Finished Training")
        print(f"Highest Validation Accuracy: {best_validation_accuracy}")

    def validate(self):
        self.model.eval().to(self.device)
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch_input_ids = batch["input_ids"].to(self.device)
                batch_attention_mask = batch["attention_mask"].to(self.device)
                batch_label = batch["label"].to(self.device)

                outputs = self.model(
                    b_input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_label,
                    return_dict=True,
                )

                loss = self.criterion(outputs.logits, batch_label)

                predictions = torch.round(torch.sigmoid(outputs.logits))
                correct_predictions += torch.sum(predictions == batch_label).item()
                total_predictions += batch_label.shape[0]
            accuracy = 100 * correct_predictions / total_predictions
        print(f"Validation Loss: {loss.item():.3f}")
        print(f"Validation Accuracy: {accuracy:.2f}%")
        return accuracy
