import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.models import Inception3


class Inception3Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Inception3(num_classes=2, transform_input=True, aux_logits=False)

        x = 3

    def forward(self, x):
        x = self.model(x.permute(0, 3, 1, 2)).logits
        x = F.softmax(x, dim=-1)
        return x

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        return opt

    def loss(self, y, y_hat):
        return F.binary_cross_entropy(y_hat, y.type(torch.float32))

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
        return loss

    def predict(self, batch, batch_idx, dataloader_idx):
        return self(batch)
