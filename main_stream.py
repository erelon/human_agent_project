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
import cv2


def create_youtube_stream(url):  # for example "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    import pafy
    play = pafy.new(url).streams[-1]  # '-1' means read the lowest quality of video.
    assert play is not None  # we want to make sure their is a input to read.
    stream = cv2.VideoCapture(play.url)  # create a opencv video stream.
    return stream


def create_webcam_stream():
    stream = cv2.VideoCapture(0)  # 0 means read from local camera.
    return stream


def main():
    transform = torchvision.transforms.Compose([
        vtransforms.StreamToTensor(fps=4, max_len=20, padding_mode="zero"),
        vtransforms.VideoResize([224, 224]),
    ])

    model = Inception3Model()
    model = model.load_from_checkpoint("95.ckpt")

    stream = create_youtube_stream("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    # stream = create_webcam_stream()
    while True:
        v = transform(stream)
        ret = model(v[None, :, :, :, :]).flatten()
        print(f'NonViolence: {ret[0]},  Violence: {ret[1]}')

    s.release()


if __name__ == '__main__':
    main()
