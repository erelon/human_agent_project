import gc

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.models import Inception3, googlenet
from torchmetrics import Accuracy


class Inception3Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.incept_out = None
        self.model = googlenet(pretrained=True, num_classes=1000, transform_input=True)
        self.model.fc.register_forward_hook(self.create_activation_hook())

        self.lstm = torch.nn.LSTM(input_size=1024, hidden_size=64, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(64, 2)

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

    def create_activation_hook(self):
        def hook(model, input, output):
            self.incept_out = input[0]

        return hook

    def forward(self, x: torch.Tensor):
        x = x.permute([0, 2, 1, 3, 4])
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)

        # The hook keeps the data
        self.model(x)
        x = self.incept_out

        x = x.view(batch_size, timesteps, -1)
        x, (h_n, h_c) = self.lstm(x)
        x = self.linear(x[:, -1, :])

        x = F.softmax(x, dim=1)

        return x

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        return opt

    def loss(self, y, y_hat):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log("train_loss", loss)
        self.log("train_acc", self.valid_acc(torch.argmax(y_hat).flatten(), y))
        torch.cuda.empty_cache()
        if batch_idx % 10 == 0:
            gc.collect(generation=2)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log("val_loss", loss)
        self.log("val_acc", self.valid_acc(torch.argmax(y_hat).flatten(), y))
        torch.cuda.empty_cache()
        if batch_idx % 10 == 0:
            gc.collect(generation=2)
        return loss

    def predict(self, batch, batch_idx, dataloader_idx):
        return self(batch)
