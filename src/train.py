import os
import warnings

import argparse
from datasets import load_dataset
import torch
import torch.nn as nn
import onnx
import mlflow
import pandas as pd
from tqdm import tqdm

from IPython.display import clear_output
from data_processing import (
    processAndAugment,
    processTestIm,
    InMemoryDataset,
    StreamingDataset,
    load_flood_train_data,
    load_flood_valid_data,
)
from SimpleUNet import SimpleUNet
from evaluation import computeAccuracy, computeIOU

# Defining global variables
LR = 5e-4
EPOCHS = 1
EPOCHS_PER_UPDATE = 1
RUNNAME = "Sen1Floods11"
S1 = "sen1floods11-dataset/v1.1/data/flood_events/HandLabeled/S1Hand/"
LABELS = "sen1floods11-dataset/v1.1/data/flood_events/HandLabeled/LabelHand/"
TRAIN_SIZE = 252


class TrainTestValidation:
    """This class contains all the metrics and methods needed
    to perform a train test validation session of SimpleUNet model."""

    # Metrics in the current epoch.
    running_loss = 0
    running_iou = 0
    running_count = 0
    running_accuracy = 0
    max_valid_iou = 0

    # Metrics lists after X epochs.
    epochs = []
    train_losses = []
    train_accuracies = []
    train_ious = []
    valid_losses = []
    valid_accuracies = []
    valid_ious = []

    # Datasets and their iterators.
    train_loader: torch.utils.data.DataLoader
    train_iter: iter

    valid_loader: torch.utils.data.DataLoader
    valid_iter: iter

    # Model parameters.
    net: SimpleUNet
    criterion: nn.CrossEntropyLoss
    optimizer: torch.optim.AdamW
    scheduler: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

    def __init__(self, s1, labels, stream=False, no_cache=False):
        if stream:
            print("STREAMING MODE ENABLED.")

            if not no_cache:
                print("CACHE IS ENABLED.")

            # Train data.
            train_data = load_dataset(
                "sen1floods11-dataset/sen1floods11_dataset.py",
                split="train",
                streaming=True,
                trust_remote_code=True,
                config_kwargs={
                    "no_cache": no_cache,
                    "context": "sen1floods11-dataset/",
                },
            )
            train_dataset = StreamingDataset(train_data, processAndAugment)
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset.with_format("numpy"),
                batch_size=16,
            )

            # Valid data.
            valid_data = load_dataset(
                "sen1floods11-dataset/sen1floods11_dataset.py",
                split="validation",
                streaming=True,
                trust_remote_code=False,
                config_kwargs={
                    "no_cache": no_cache,
                    "context": "sen1floods11-dataset/",
                },
            )
            valid_dataset = StreamingDataset(valid_data, processTestIm)
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset.with_format("numpy"),
                batch_size=4,
                collate_fn=lambda x: (
                    torch.cat([a[0] for a in x], 0),
                    torch.cat([a[1] for a in x], 0),
                ),
            )
        else:
            print("STREAMING MODE DISABLED.")
            train_data = load_flood_train_data(s1, labels)
            train_dataset = InMemoryDataset(train_data, processAndAugment)
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=16,
                shuffle=True,
                sampler=None,
                batch_sampler=None,
                num_workers=0,
                collate_fn=None,
                pin_memory=False,
                drop_last=False,
                timeout=0,
                worker_init_fn=None,
            )

            valid_data = load_flood_valid_data(s1, labels)
            valid_dataset = InMemoryDataset(valid_data, processTestIm)
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=4,
                shuffle=True,
                sampler=None,
                batch_sampler=None,
                num_workers=0,
                collate_fn=lambda x: (
                    torch.cat([a[0] for a in x], 0),
                    torch.cat([a[1] for a in x], 0),
                ),
                pin_memory=False,
                drop_last=False,
                timeout=0,
                worker_init_fn=None,
            )

        self.valid_iter = iter(self.valid_loader)
        self.train_iter = iter(self.train_loader)

        self.net = SimpleUNet()
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1, 8]).float(), ignore_index=255
        )
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=LR)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            TRAIN_SIZE * 10,
            T_mult=2,
            eta_min=0,
            last_epoch=-1,
        )

    def train_and_valid(self):
        """Run the training and validating session of the SimpleUNet moddel."""
        for i in range(0, EPOCHS):
            self.train_validation_loop(num_epochs=10, cur_epoch=i)
            self.epochs.append(i)
            print("max valid iou:", self.max_valid_iou)

    def train_validation_loop(
        self,
        num_epochs,
        cur_epoch,
    ):
        """Train and validation loop for 1 epoch."""

        self.net = self.net.train()
        self.running_loss = 0
        self.running_iou = 0
        self.running_count = 0
        self.running_accuracy = 0

        for _ in range(num_epochs):
            self.train_epoch()
        clear_output()

        print("Current Epoch:", cur_epoch)
        self.validation_loop(cur_epoch)

    def train_epoch(self):
        """Train for 1 epoch."""
        self.train_iter = iter(self.train_loader)
        for inputs, labels in tqdm(self.train_iter, desc="Training Loop"):
            self.train_loop(inputs, labels)

    def train_loop(self, inputs, labels):
        """Train in 1 picture."""
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # net = net.cuda()

        # forward + backward + optimize
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels.long())
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.running_loss += loss
        self.running_iou += computeIOU(outputs, labels)
        self.running_accuracy += computeAccuracy(outputs, labels)
        self.running_count += 1

    def validation_loop(self, epoch):
        """Validation for 1 epoch."""
        self.net = self.net.eval()
        # net = net.cuda()
        count = 0
        iou = 0
        loss = 0
        accuracy = 0
        self.valid_iter = iter(self.valid_loader)
        with torch.no_grad():
            for images, labels in self.valid_iter:
                # net = net.cuda()
                outputs = self.net(images)
                valid_loss = self.criterion(outputs, labels.long())
                valid_iou = computeIOU(outputs, labels)
                valid_accuracy = computeAccuracy(outputs, labels)
                iou += valid_iou
                loss += valid_loss
                accuracy += valid_accuracy
                count += 1

        iou = iou / count
        accuracy = accuracy / count

        if iou > self.max_valid_iou:
            self.max_valid_iou = iou
            save_path = os.path.join(
                "checkpoints", "{}_{}_{}.onnx".format(RUNNAME, epoch, iou.item())
            )
            # torch.save(self.net.state_dict(), save_path)
            input_tensor = torch.randn(4, 2, 256, 256)
            torch.onnx.export(self.net, input_tensor, save_path)
            print("model saved at", save_path, ".")

            onnx_model = onnx.load(save_path)
            artifact_path = f"iou_{iou.item()}"
            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                artifact_path=artifact_path,
                input_example=input_tensor.numpy(),
            )
            print("model saved in MLflow.")

        loss = loss / count
        mlflow.log_metric("loss", loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)
        mlflow.log_metric("IOU", iou, step=epoch)
        print("Training Loss:", self.running_loss / self.running_count)
        print("Training IOU:", self.running_iou / self.running_count)
        print("Training Accuracy:", self.running_accuracy / self.running_count)
        print("Validation Loss:", loss)
        print("Validation IOU:", iou)
        print("Validation Accuracy:", accuracy)

        self.train_losses.append(self.running_loss / self.running_count)
        self.train_accuracies.append(self.running_accuracy / self.running_count)
        self.train_ious.append(self.running_iou / self.running_count)
        self.valid_losses.append(loss)
        self.valid_accuracies.append(accuracy)
        self.valid_ious.append(iou)


def run():
    """Run the training of the SimpleUNet model."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream", action="store_true", help="Streaming mode (by default: False)"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Cache mode (by default: True)"
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    mlflow.set_tracking_uri(
        "https://sharinghub.p2.csgroup.space/mlflow/space_applications/mlops-services/sample-projects/ai-models/flood-model/tracking/"
    )
    mlflow.set_experiment("first_experiment (1493)")

    with mlflow.start_run():

        mlflow.log_param("dvc_file_path", "v1.1.dvc")
        mlflow.log_param("LR", LR)
        mlflow.log_param("EPOCH_PER_UPDATE", EPOCHS_PER_UPDATE)

        dataset_source_url = "https://gitlab.si.c-s.fr/space_applications/mlops-services/sample-projects/datasets/sen1floods11-dataset"
        raw_data = pd.read_csv(
            dataset_source_url,
            delimiter=",",
            on_bad_lines="skip",
        )

        dataset = mlflow.data.from_pandas(
            raw_data,
            source=dataset_source_url,
            name="flood train data",
        )

        mlflow.log_input(dataset, context="training")

        tv = TrainTestValidation(S1, LABELS, stream=args.stream, no_cache=args.no_cache)

        tv.train_and_valid()


if __name__ == "__main__":
    run()