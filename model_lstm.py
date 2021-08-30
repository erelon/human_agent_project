from torchvision.models import googlenet
from torchmetrics import Accuracy, AUROC
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import gc

from mlstmfcn import MLSTMfcn


class Inception3Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.incept_out = None
        self.model_f = googlenet(pretrained=True, num_classes=1000, transform_input=True)
        self.model_f.fc.register_forward_hook(self.create_activation_hook())
        # self.model.features.register_forward_hook(self.create_activation_hook())

        # self.lstm = torch.nn.LSTM(input_size=1024, hidden_size=128, num_layers=5, batch_first=True)
        # self.linear = torch.nn.Linear(128, 2)
        self.model = MLSTMfcn(num_classes=2, max_seq_len=40, num_features=1024)
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.valid_auc = AUROC(num_classes=2)

    def create_activation_hook(self):
        def hook(model, input, output):
            self.incept_out = input[0]

        return hook

    def forward(self, x: torch.Tensor):
        """

        :param x: Tensor of the shape (Batch, Channels, Timestamp(Length), Height, Width)
        :return: Tensor with prediction for V/NV
        """
        x = x.permute([0, 2, 1, 3, 4])
        batch_size, timesteps, C, H, W = x.size()

        # The hook keeps the data
        x = x.reshape(batch_size * timesteps, C, H, W)
        self.model_f(x)
        x = self.incept_out

        x = x.reshape(batch_size, timesteps, -1)
        x = self.model(x, [timesteps for i in range(batch_size)])

        # x = x.view(batch_size, timesteps, -1)
        # x, (h_n, h_c) = self.lstm(x)
        # x = self.linear(x[:, -1, :])

        # x = F.softmax(x, dim=1)

        return x

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=5e-4)
        return opt

    def loss(self, y, y_hat):
        # This is Cross-Entropy loss
        return F.nll_loss(torch.log(y_hat), y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log("train_loss", loss)

        self.train_acc(torch.argmax(y_hat, dim=1).flatten(), y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        torch.cuda.empty_cache()
        if batch_idx % 10 == 0:
            gc.collect(generation=2)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y, y_hat)
        self.log("val_loss", loss)

        self.valid_acc(torch.argmax(y_hat, dim=1).flatten(), y)
        self.log("val_acc", self.valid_acc, on_step=True, on_epoch=True)
        try:
            self.valid_auc(y_hat, y)
            self.log("val_auc", self.valid_auc, on_step=True, on_epoch=True)
        except Exception as ex:
            print(ex)

        torch.cuda.empty_cache()
        if batch_idx % 10 == 0:
            gc.collect(generation=2)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        tag = torch.argmax(y_hat, dim=1)
        print(tag, y)
        return y_hat
