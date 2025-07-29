import importlib.util
import random
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms


def load_dataset(
    dataset_path: Path,
) -> Mapping[str, Callable[..., Generator[dict, Any, Any]]]:
    # sys.path.append(str(dataset_path))
    # from dataset import DatasetLoader
    # return DatasetLoader(dataset_path)

    module_path = dataset_path / "dataset.py"
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    DatasetLoader = getattr(module, "DatasetLoader")
    return DatasetLoader(dataset_path)


def process_training_data(data: dict) -> dict:
    image, mask = _prepare_data(data)

    im, label = image.copy(), mask.copy()

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


def process_validation_data(data: dict) -> dict:
    image, mask = _prepare_data(data)

    im, label = image.copy(), mask.copy()
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


def _prepare_data(data: dict) -> tuple[np.ndarray, np.ndarray]:
    image, mask = data["image"], data["mask"]
    image = _prepare_image_data(image)
    mask = _prepare_mask_data(mask)
    return image, mask


def _prepare_image_data(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image)
    image = np.clip(image, -50, 1)
    image = (image + 50) / 51
    return image


def _prepare_mask_data(mask: np.ndarray) -> np.ndarray:
    mask[mask == -1] = 255
    # mask = mask.astype(np.int16)
    return mask
