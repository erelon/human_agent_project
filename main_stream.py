import video_dataloader.transforms as vtransforms
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import matplotlib as mpl
import torchvision
import numpy as np
import torch
import tqdm
import sys
import os

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAMPlusPlus as GradCAM
from video_parse import create_path_csv
from torch.utils.data import DataLoader
from model_lstm import Inception3Model
from video_dataloader import datasets
from matplotlib import animation
from datetime import datetime
import cv2


def create_youtube_stream(url):  # for example "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    import pafy
    play = pafy.new(url).streams[-1]  # '-1' means read the lowest quality of video.
    assert play is not None  # we want to make sure their is a input to read.
    stream = cv2.VideoCapture(play.url)  # create a opencv video stream.
    return stream


def create_webcam_stream(cam_id):
    stream = cv2.VideoCapture(cam_id)  # 0 means read from local camera.
    return stream


def main():
    stream = None
    if len(sys.argv) == 2:
        url = sys.argv[1].strip()
        if url.lower().startswith('webcam'):
            url = url.replace('webcam', '')
            cam_id = int(url) if url.isdigit() else 0
            print('using webcam')
            stream = create_webcam_stream(cam_id)
        elif 'http' in url.lower() and 'www.youtube.com' in url.lower():
            print(f'using youtube url: {url}')
            stream = create_youtube_stream(url)

    if stream is None:
        print('USAGE: main_stream.py youtube_url/webcam')
        sys.exit(0)

    stt = vtransforms.StreamToTensor(stream=stream, fps=4, max_len=20, padding_mode="zero")
    transform = torchvision.transforms.Compose([
        stt,
        vtransforms.VideoResize([224, 224]),
    ])

    model = Inception3Model()
    model = model.load_from_checkpoint("95.ckpt")
    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    model.eval()
    with torch.no_grad():
        while stt.is_running:
            v = transform(None)
            ret = model(v[None, :, :, :, :]).flatten()

            print(f'NonViolence: {ret[0]},  Violence: {ret[1]}')

    stream.release()


if __name__ == '__main__':
    main()
