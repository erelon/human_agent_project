from datetime import datetime

import video_dataloader.transforms as vtransforms
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import matplotlib as mpl
import torchvision
import numpy as np
import torch
import tqdm
import os

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAMPlusPlus as GradCAM
from video_parse import create_path_csv
from torch.utils.data import DataLoader
from model_lstm import Inception3Model
from video_dataloader import datasets
from matplotlib import animation

if __name__ == '__main__':
    path_train, path_test = create_path_csv()
    # path_train, path_test = "paths_train.csv", "paths_test.csv"

    train_videos = datasets.VideoLabelDataset(
        path_train,
        transform=torchvision.transforms.Compose([
            vtransforms.VideoFilePathToTensor(fps=4, max_len=20, padding_mode="zero"),
            # vtransforms.VideoRandomCrop([512, 512]),
            vtransforms.VideoResize([224, 224]),
        ])
    )
    test_videos = datasets.VideoLabelDataset(
        path_test,
        transform=torchvision.transforms.Compose([
            vtransforms.VideoFilePathToTensor(fps=4, max_len=20, padding_mode="zero"),
            # vtransforms.VideoRandomCrop([512, 512]),
            vtransforms.VideoResize([224, 224]),
        ])
    )

    if torch.cuda.is_available():
        data_loader_train = torch.utils.data.DataLoader(train_videos, batch_size=4, shuffle=True, num_workers=8)
        data_loader_test = torch.utils.data.DataLoader(test_videos, batch_size=4, shuffle=False, num_workers=8)
        t = pl.Trainer(gpus=[3])
    else:
        data_loader_train = torch.utils.data.DataLoader(train_videos, batch_size=2, shuffle=True)
        data_loader_test = torch.utils.data.DataLoader(test_videos, batch_size=2, shuffle=False)
        t = pl.Trainer()

    model = Inception3Model()
    model = model.load_from_checkpoint("95.ckpt")
    # t.fit(model, train_dataloader=data_loader_train, val_dataloaders=[data_loader_test])

    # all_preds = t.predict(model, data_loader_test, return_predictions=True)
    # print(all_preds)

    k = 0

    mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Program Files\\Softdeluxe\\Free Download Manager\\ffmpeg.exe'
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=2, bitrate=1800)

    grad_cam = GradCAM(model, model.model_f.inception5b)
    for vid in data_loader_train:
        vids, y = vid
        for v, label in zip(vids, y):
            grads_v = grad_cam(v.unsqueeze(0), target_category=1)
            grads_nv = grad_cam(v.unsqueeze(0), target_category=0)
            pred = model(v.unsqueeze(0)).flatten()
            v = v.permute(1, 2, 3, 0)
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
            ims = []
            for im, gradv, gradnv in zip(v, grads_v, grads_nv):
                im = im.numpy()
                visualization_v = show_cam_on_image(im, gradv) / 255
                visualization_nv = show_cam_on_image(im, gradnv) / 255
                ims.append([ax1.imshow(visualization_nv, animated=True), ax2.imshow(im, animated=True),
                            ax3.imshow(visualization_v, animated=True)])
            ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True,
                                            repeat_delay=100)
            ax1.set_title("grads NonViolence")
            ax2.set_title("Original video")
            ax3.set_title("grads Violence")

            ax1.set_xlabel(f"P\n{round(pred[0].item(), 3)}")
            ax2.set_xlabel(f"GT\n{'Violence' if label == 1 else 'NonViolence'}")
            ax3.set_xlabel(f"P\n{round(pred[1].item(), 3)}")
            # plt.axis('off')

            for axs in fig.axes:
                axs.get_xaxis().set_ticks([])
                axs.get_yaxis().set_ticks([])

            ani.save(f'GradCam/{k}.mp4', writer=writer)
            k += 1
            # plt.show(block=True)
            # fig.clf()
            plt.close(fig)
