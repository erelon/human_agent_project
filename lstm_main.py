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
    path_csv = create_path_csv()

    all_videos = datasets.VideoLabelDataset(
        path_csv,
        transform=torchvision.transforms.Compose([
            vtransforms.VideoFilePathToTensor(fps=4),
            # vtransforms.VideoRandomCrop([512, 512]),
            vtransforms.VideoResize([299, 299]),
        ])
    )

    train_size = int(len(all_videos) * 0.7)
    train, test = torch.utils.data.random_split(all_videos, [train_size, len(all_videos) - train_size])

    data_loader_train = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

    model = Inception3Model()
    t = pl.Trainer()
    t.fit(model, data_loader_train, data_loader_test)
