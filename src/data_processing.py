import csv
import os
import random

import mlflow
import numpy as np
import rasterio
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms


class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])

    def __len__(self):
        return len(self.data_list)


class StreamingDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_iterable, preprocess_func):
        self.data_iterable = data_iterable
        self.preprocess_func = preprocess_func

    def __iter__(self):
        for data in self.data_iterable:
            yield self.preprocess_func(data, streaming=True)

    def with_format(self, format_type):
        self.data_iterable = self.data_iterable.with_format(format_type)
        return self


def processAndAugment(data, streaming=False):
    if streaming:
        x = data["image"]
        y = data["mask"].astype(np.int16)
    else:
        (x, y) = data
    im, label = x.copy(), y.copy()

    # convert to PIL for easier transforms
    im1 = Image.fromarray(im[0])
    im2 = Image.fromarray(im[1])
    label = Image.fromarray(label.squeeze())

    # Get params for random transforms
    i, j, h, w = transforms.RandomCrop.get_params(im1, (256, 256))

    im1 = F.crop(im1, i, j, h, w)
    im2 = F.crop(im2, i, j, h, w)
    label = F.crop(label, i, j, h, w)
    if random.random() > 0.5:
        im1 = F.hflip(im1)
        im2 = F.hflip(im2)
        label = F.hflip(label)
    if random.random() > 0.5:
        im1 = F.vflip(im1)
        im2 = F.vflip(im2)
        label = F.vflip(label)

    norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
    im = torch.stack(
        [transforms.ToTensor()(im1).squeeze(), transforms.ToTensor()(im2).squeeze()]
    )
    im = norm(im)
    label = transforms.ToTensor()(label).squeeze()
    if torch.sum(label.gt(0.003) * label.lt(0.004)):
        label *= 255
    label = label.round()

    return im, label


def processTestIm(data, streaming=False):
    if streaming:
        x = data["image"]
        y = data["mask"].astype(np.int16)
    else:
        (x, y) = data
    im, label = x.copy(), y.copy()
    norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])

    # convert to PIL for easier transforms
    im_c1 = Image.fromarray(im[0]).resize((512, 512))
    im_c2 = Image.fromarray(im[1]).resize((512, 512))
    label = Image.fromarray(label.squeeze()).resize((512, 512))

    im_c1s = [
        F.crop(im_c1, 0, 0, 256, 256),
        F.crop(im_c1, 0, 256, 256, 256),
        F.crop(im_c1, 256, 0, 256, 256),
        F.crop(im_c1, 256, 256, 256, 256),
    ]
    im_c2s = [
        F.crop(im_c2, 0, 0, 256, 256),
        F.crop(im_c2, 0, 256, 256, 256),
        F.crop(im_c2, 256, 0, 256, 256),
        F.crop(im_c2, 256, 256, 256, 256),
    ]
    labels = [
        F.crop(label, 0, 0, 256, 256),
        F.crop(label, 0, 256, 256, 256),
        F.crop(label, 256, 0, 256, 256),
        F.crop(label, 256, 256, 256, 256),
    ]

    ims = [
        torch.stack(
            (transforms.ToTensor()(x).squeeze(), transforms.ToTensor()(y).squeeze())
        )
        for (x, y) in zip(im_c1s, im_c2s)
    ]

    ims = [norm(im) for im in ims]
    ims = torch.stack(ims)

    labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
    labels = torch.stack(labels)

    if torch.sum(labels.gt(0.003) * labels.lt(0.004)):
        labels *= 255
    labels = labels.round()

    return ims, labels


def getArrFlood(fname):
    return rasterio.open(fname).read()


def download_flood_water_data_from_list(l):
    i = 0
    flood_data = []
    for im_fname, mask_fname in l:
        if not os.path.exists(im_fname):
            continue
        arr_x = np.nan_to_num(getArrFlood(im_fname))
        arr_y = getArrFlood(mask_fname)
        arr_y[arr_y == -1] = 255

        arr_x = np.clip(arr_x, -50, 1)
        arr_x = (arr_x + 50) / 51

        if i % 100 == 0:
            print(im_fname, mask_fname)
        i += 1
        flood_data.append((arr_x, arr_y))

    return flood_data


def load_flood_train_data(input_root, label_root):
    fname = "sen1floods11-dataset/flood_train_data.csv"
    mlflow.log_artifact(fname, artifact_path="train_data")
    training_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            training_files.append(tuple((input_root + line[0], label_root + line[1])))

    return download_flood_water_data_from_list(training_files)


def load_flood_test_data(input_root, label_root):
    fname = "sen1floods11-dataset/flood_test_data.csv"
    testing_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            testing_files.append(tuple((input_root + line[0], label_root + line[1])))

    return download_flood_water_data_from_list(testing_files)


def load_flood_valid_data(input_root, label_root):
    fname = "sen1floods11-dataset/flood_valid_data.csv"
    mlflow.log_artifact(fname, artifact_path="valid_data")
    validation_files = []
    with open(fname) as f:
        for line in csv.reader(f):
            validation_files.append(tuple((input_root + line[0], label_root + line[1])))

    return download_flood_water_data_from_list(validation_files)
