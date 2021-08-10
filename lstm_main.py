import os

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
    if "path.csv" not in os.listdir():
        path_csv = create_path_csv()
    else:
        path_csv = "path.csv"

    all_videos = datasets.VideoLabelDataset(
        path_csv,
        transform=torchvision.transforms.Compose([
            vtransforms.VideoFilePathToTensor(fps=4, max_len=100),
            # vtransforms.VideoRandomCrop([512, 512]),
            vtransforms.VideoResize([224, 224]),
        ])
    )

    train_size = int(len(all_videos) * 0.7)
    train, test = torch.utils.data.random_split(all_videos, [train_size, len(all_videos) - train_size])

    if torch.cuda.is_available():
        data_loader_train = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True, num_workers=8)
        data_loader_test = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False, num_workers=8)
        t = pl.Trainer(gpus=[3])
    else:
        data_loader_train = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
        data_loader_test = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)
        t = pl.Trainer()

    model = Inception3Model()
    t.fit(model, data_loader_train, data_loader_test)
