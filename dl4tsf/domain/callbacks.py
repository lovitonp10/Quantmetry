from pytorch_lightning.callbacks import Callback


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.losses = {"train_loss": self.train_losses, "val_loss": self.val_losses}
        self.metrics = {"loss": self.losses}

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics["train_loss"].item()
        self.train_losses.append(train_loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(val_loss)
