import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.models import Inception3


class Inception3Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.incept_out = None
        self.model = Inception3(num_classes=2, transform_input=True)
        self.model.fc.register_forward_hook(self.create_activation_hook())

        self.lstm = torch.nn.LSTM(input_size=2048, hidden_size=64, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(64, 2)

    def create_activation_hook(self):
        def hook(model, input, output):
            self.incept_out = input[0]

        return hook

    def forward(self, x: torch.Tensor):
        x = x.permute([0, 2, 1, 3, 4])
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)

        # The hook keeps the data
        self.model(c_in)
        c_out = self.incept_out

        r_in = c_out.view(batch_size, timesteps, -1)
        x, (h_n, h_c) = self.lstm(r_in)
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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log("val_loss", loss)
        self.log("val_acc", torch.argmax(y_hat) == y)
        return loss

    def predict(self, batch, batch_idx, dataloader_idx):
        return self(batch)
