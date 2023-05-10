import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from tqdm import tqdm


class Trainer:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.criterion = binary_cross_entropy_with_logits

    def run(self):
        print(f"Training started on {self.DEVICE}.")
        for _epoch in range(self.epochs):
            self.train()
            self.validate()

    def train(self):
        self.model.to(self.DEVICE)
        self.model.train()
        best_validation_accuracy = 0

        for epoch in range(1, self.epochs):
            loop = tqdm(self.train_dataloader)
            for batch in loop:
                batch_input_ids = batch["input_ids"].to(self.DEVICE)
                batch_attention_mask = batch["attention_mask"].to(self.DEVICE)
                batch_label = batch["label"].to(self.DEVICE)

                self.optimizer.zero_grad()

                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_label,
                    return_dict=True,
                )

                batch_label = batch_label.to(torch.float).unsqueeze(1)
                loss = self.criterion(outputs.logits, batch_label)
                loss.backward()
                self.optimizer.step()

                loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item())

            print(f"Training Loss: {loss.item():.5f}")

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
        self.model.to(self.DEVICE)
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch_input_ids = batch["input_ids"].to(self.DEVICE)
                batch_attention_mask = batch["attention_mask"].to(self.DEVICE)
                batch_label = batch["label"].to(self.DEVICE)

                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_label,
                    return_dict=True,
                )

                batch_label = batch_label.to(torch.float).unsqueeze(1)
                loss = self.criterion(outputs.logits, batch_label)

                predictions = torch.round(torch.sigmoid(outputs.logits))
                correct_predictions += torch.sum(predictions == batch_label).item()
                total_predictions += batch_label.shape[0]
            accuracy = 100 * correct_predictions / total_predictions

        print(f"Validation Loss: {loss.item():.5f}")
        print(f"Validation Accuracy: {accuracy:.2f}%")
        return accuracy
