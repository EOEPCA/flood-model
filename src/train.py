#!/usr/bin/env python3
"""This file can be used to perform training for the flood detection model."""

import argparse
import os
import time
import uuid
from pathlib import Path

import datasets
import mlflow
import onnx
import torch
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from tqdm import tqdm

from data_processing import load_dataset, process_training_data, process_validation_data
from evaluation import computeAccuracy, computeIOU
from model import SimpleUNet


class FloodModelTraining:
    def __init__(
        self,
        output_path: Path,
        dataset_path: Path,
        lr: float,
        epochs: int = 1,
        stream: bool = False,
        no_cache: bool = False,
    ) -> None:
        self.model_path = output_path

        # Parameters
        self.lr = lr
        self.epochs = epochs

        # Metrics for X epochs.
        self.max_valid_iou = 0

        # Metrics for current epoch.
        self.epoch_loss = 0
        self.epoch_iou = 0
        self.epoch_accuracy = 0
        self.epoch_batch_count = 0

        # Dataset
        dataset_loader = load_dataset(dataset_path)

        _train_dataset = dataset_loader["train"]
        self.train_dataset = datasets.IterableDataset.from_generator(
            _train_dataset,
            gen_kwargs={
                "shuffle": True,
                "stream": stream,
                "stream_cache": not no_cache,
                "process_func": process_training_data,
            },
        )
        self.train_size = len(_train_dataset)

        _valid_dataset = dataset_loader["validation"]
        self.valid_dataset = datasets.IterableDataset.from_generator(
            _valid_dataset,
            gen_kwargs={
                "stream": stream,
                "stream_cache": not no_cache,
                "process_func": process_validation_data,
            },
        )
        self.valid_size = len(_valid_dataset)

        # Model
        self.net = SimpleUNet()
        self._criterion = torch.torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1, 8]).float(), ignore_index=255
        )
        self._optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self._optimizer,
            T_0=self.train_size * 10,
            T_mult=2,
            eta_min=0,
            last_epoch=-1,
        )

    def run(self) -> Path | None:
        print("Training Flood Model...")

        mlflow.log_param("LR", self.lr)
        mlflow.log_param("EPOCHS", self.epochs)

        for i in range(self.epochs):
            self._run_epoch(epoch=i)

        if self.model_path.is_file():
            onnx_model = onnx.load(self.model_path)
            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                artifact_path="model",
                input_example=torch.randn(4, 2, 256, 256).numpy(),
                save_as_external_data=False,
            )
            print("MAX IOU:", self.max_valid_iou)
            print("Model at", self.model_path)
            print("MLflow: model logged")
            return self.model_path

        return None

    def _run_epoch(self, epoch: int = 0) -> None:
        print(f"[EPOCH {epoch}] START")

        self.epoch_loss = 0
        self.epoch_iou = 0
        self.epoch_accuracy = 0
        self.epoch_batch_count = 0

        self._run_train(epoch)
        self._run_validation(epoch)

        print(f"[EPOCH {epoch}] END")

    def _run_train(self, epoch: int) -> None:
        print(f"[EPOCH {epoch}] Training")
        self.net = self.net.train()

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=16,
        )
        for inputs, labels in tqdm(
            iter(train_loader), desc="Training Loop", total=train_loader.batch_size
        ):
            self._train_loop(inputs, labels)

    def _train_loop(self, inputs: torch.Tensor, labels: torch.Tensor):
        # zero the parameter gradients
        self._optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.net(inputs)
        loss = self._criterion(outputs, labels.long())
        loss.backward()
        self._optimizer.step()
        self._scheduler.step()

        self.epoch_loss += loss
        self.epoch_iou += computeIOU(outputs, labels)
        self.epoch_accuracy += computeAccuracy(outputs, labels)
        self.epoch_batch_count += 1

    def _run_validation(self, epoch: int) -> None:
        print(f"[EPOCH {epoch}] Validation")
        self.net = self.net.eval()

        count = 0
        iou = 0
        loss = 0
        accuracy = 0
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=4,
            collate_fn=lambda x: (
                torch.cat([a[0] for a in x], 0),
                torch.cat([a[1] for a in x], 0),
            ),
        )

        with torch.no_grad():
            for inputs, labels in tqdm(
                iter(valid_loader),
                desc="Validation Loop",
                total=self.valid_size // valid_loader.batch_size
                + self.valid_size % valid_loader.batch_size,
            ):
                outputs = self.net(inputs)
                valid_loss = self._criterion(outputs, labels.long())
                valid_iou = computeIOU(outputs, labels)
                valid_accuracy = computeAccuracy(outputs, labels)
                iou += valid_iou
                loss += valid_loss
                accuracy += valid_accuracy
                count += 1

        iou = iou / count
        accuracy = accuracy / count
        loss = loss / count

        mlflow.log_metric("loss", loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)
        mlflow.log_metric("IOU", iou, step=epoch)

        print(
            f"[EPOCH {epoch}] Training Loss:", self.epoch_loss / self.epoch_batch_count
        )
        print(f"[EPOCH {epoch}] Training IOU:", self.epoch_iou / self.epoch_batch_count)
        print(
            f"[EPOCH {epoch}] Training Accuracy:",
            self.epoch_accuracy / self.epoch_batch_count,
        )
        print(f"[EPOCH {epoch}] Validation Loss:", loss)
        print(f"[EPOCH {epoch}] Validation IOU:", iou)
        print(f"[EPOCH {epoch}] Validation Accuracy:", accuracy)

        if iou > self.max_valid_iou:
            self.max_valid_iou = iou

            if not self.model_path.parent.exists():
                self.model_path.parent.mkdir(parents=True, exist_ok=True)

            torch.onnx.export(
                self.net,
                torch.randn(4, 2, 256, 256),
                self.model_path,
            )
            print(f"[EPOCH {epoch}] Best IOU: model saved")


def main():
    """Run the training of the SimpleUNet model."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate of the training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Epochs of the training",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        help="Path of the output model",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="sen1floods11-dataset",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--dataset-url",
        # default="https://gitlab.develop.eoepca.org/sharinghub-test/sen1floods11-dataset",
        help="URL of the dataset",
    )
    parser.add_argument(
        "--dataset-version",
        help="Version of the dataset",
    )
    parser.add_argument(
        "--stream", action="store_true", help="Streaming mode (by default: disabled)"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Cache mode (by default: enabled)"
    )
    parser.add_argument(
        "--mlflow-uri",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="Training",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.is_dir():
        print(f"ERROR: Dataset not found at '{dataset_path}'")

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    fmt = FloodModelTraining(
        output_path=Path(args.output_path)
        if args.output_path
        else Path(
            "checkpoints", f"model_{int(time.time())}_{str(uuid.uuid4())[:8]}.onnx"
        ),
        dataset_path=dataset_path,
        epochs=args.epochs,
        lr=args.learning_rate,
        stream=args.stream,
        no_cache=args.no_cache,
    )
    with mlflow.start_run():
        if docker_image := os.environ.get("IMAGE_NAME"):
            mlflow.set_tag("Docker image", docker_image, synchronous=True)
            print("MLflow: Docker image logged")

        if args.dataset_url:
            dataset_metadata = MetaDataset(
                source=HTTPDatasetSource(url=args.dataset_url),
                name="Sen1floods11 Dataset",
                digest=args.dataset_version,
            )
            mlflow.log_input(dataset_metadata)
            print("MLflow: Dataset metadata logged")

        if args.stream:
            mlflow.set_tag("Data", "stream", synchronous=True)
        else:
            mlflow.set_tag("Data", "local", synchronous=True)

        fmt.run()


if __name__ == "__main__":
    main()
