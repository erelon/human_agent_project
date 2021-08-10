import os

import tqdm

import video_dataloader.transforms as vtransforms
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchvision
import torch

from video_parse import create_path_csv
from torch.utils.data import DataLoader
from model_lstm import Inception3Model
from video_dataloader import datasets

if __name__ == '__main__':
    path_train, path_test = create_path_csv()

    train_videos = datasets.VideoLabelDataset(
        path_train,
        transform=torchvision.transforms.Compose([
            vtransforms.VideoFilePathToTensor(fps=4, max_len=20, padding_mode="zero"),
            # vtransforms.VideoRandomCrop([512, 512]),
            vtransforms.VideoResize([64, 64]),
        ])
    )
    test_videos = datasets.VideoLabelDataset(
        path_test,
        transform=torchvision.transforms.Compose([
            vtransforms.VideoFilePathToTensor(fps=4, max_len=20, padding_mode="zero"),
            # vtransforms.VideoRandomCrop([512, 512]),
            vtransforms.VideoResize([64, 64]),
        ])
    )

    if torch.cuda.is_available():
        data_loader_train = torch.utils.data.DataLoader(train_videos, batch_size=4, shuffle=True, num_workers=8)
        data_loader_test = torch.utils.data.DataLoader(test_videos, batch_size=4, shuffle=False, num_workers=8)
        t = pl.Trainer(gpus=[3])
    else:
        data_loader_train = torch.utils.data.DataLoader(train_videos, batch_size=4, shuffle=True)
        data_loader_test = torch.utils.data.DataLoader(test_videos, batch_size=2, shuffle=False)
        t = pl.Trainer()

    model = Inception3Model()
    # model.load_from_checkpoint("what.ckpt")
    t.fit(model, train_dataloader=data_loader_train, val_dataloaders=[data_loader_test])

    # all_preds = t.predict(model, data_loader_test, return_predictions=True)
    # print(all_preds)
