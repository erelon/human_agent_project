import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import torch
import cv2
import os

from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from model import Inception3Model
from tqdm import tqdm

VIOLENCE_LABEL = 1
NONVIOLENCE_LABEL = 0


def load_images(video_path):
    # os.makedirs('ViolenceImgs', exist_ok=True)
    all_imgs = []
    for path in tqdm(sorted(os.listdir(video_path))):
        vidcap = cv2.VideoCapture(f"{video_path}/{path}")
        success, image = vidcap.read()
        count = 0
        imgs = []
        while success:
            if count % 5 == 0:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (299, 299))
                imgs.append(image)
            success, image = vidcap.read()
            count += 1
        all_imgs.append(np.array(imgs))
    return all_imgs


if __name__ == '__main__':
    all_vilonece_images = load_images("Violence")
    all_nonvilonece_images = load_images("NonViolence")

    main_dataset = []
    for vid in all_vilonece_images:
        num_of_imgs_in_vid = vid.shape[0]
        labels = torch.tensor(np.zeros([num_of_imgs_in_vid, 2]))
        labels[:, VIOLENCE_LABEL] = 1
        vid_dataset = TensorDataset(torch.tensor(vid), labels)
        main_dataset.append(vid_dataset)
    for vid in all_nonvilonece_images:
        num_of_imgs_in_vid = vid.shape[0]
        labels = torch.tensor(np.zeros([num_of_imgs_in_vid, 2]))
        labels[:, NONVIOLENCE_LABEL] = 1
        vid_dataset = TensorDataset(torch.tensor(vid), labels)

        main_dataset.append(vid_dataset)

    train, test = train_test_split(main_dataset, test_size=0.25)

    train_dataset = ConcatDataset(train)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    valid_dataset = ConcatDataset(test)
    valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False)

    # zoom_range = 0.15
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomRotation(30),
    #     transforms.RandomAffine(0, shear=0.15, scale=(1 + zoom_range, 1 - zoom_range), translate=(0.2, 0.2)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.PILToTensor()
    # ])

    model = Inception3Model()
    t = pl.Trainer()
    t.fit(model, train_dataloader)
