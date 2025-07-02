"""This file can be used to perform an inference on the trained flood detection model."""

import argparse
import datetime
import glob
import mimetypes
import os
import uuid
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pystac
import rasterio
from PIL import Image


def load_model(fpath):
    """Load the model in evaluation mode."""

    ort_session = ort.InferenceSession(fpath)
    return ort_session


def get_arr_flood(fpath):
    """open image with rasterio."""
    return rasterio.open(fpath).read()


def prepare_input_inference(fpath):
    """prepare input."""
    arr_x = np.nan_to_num(get_arr_flood(fpath))
    arr_x = np.clip(arr_x, -50, 1)
    arr_x = (arr_x + 50) / 51
    return arr_x


def normalize(image, mean, std):
    """
    Normalizes an image using the provided mean and standard deviation.
    Args:
    - image (np.ndarray): The image to normalize, shape (C, H, W).
    - mean (list or np.ndarray): The mean for each channel.
    - std (list or np.ndarray): The standard deviation for each channel.

    Returns:
    - np.ndarray: The normalized image.
    """
    for c in range(image.shape[0]):
        image[c] = (image[c] - mean[c]) / std[c]
    return image


def to_numpy(pil_image):
    """
    Converts a PIL image to a NumPy array with shape (C, H, W).
    Args:
    - pil_image (PIL.Image): The PIL image to convert.

    Returns:
    - np.ndarray: The converted image.
    """
    return np.array(pil_image, dtype=np.float32)


def crop(im, top, left, height, width):
    """Crop into four patches."""
    return im.crop((left, top, left + width, top + height))


def process_inference(image):
    """
    Preprocesses an input image for an ONNX model.
    Args:
    - image (np.ndarray): Input image with shape (C, H, W).

    Returns:
    - np.ndarray: Preprocessed images ready for the model, shape (N, C, H, W).
    """
    im = image.copy()

    # Mean and standard deviation for normalization
    mean = [0.6851, 0.5235]
    std = [0.0820, 0.1102]

    # Convert to PIL images for transformations
    im_c1 = Image.fromarray(im[0]).resize((512, 512))
    im_c2 = Image.fromarray(im[1]).resize((512, 512))

    im_c1s = [
        crop(im_c1, 0, 0, 256, 256),
        crop(im_c1, 0, 256, 256, 256),
        crop(im_c1, 256, 0, 256, 256),
        crop(im_c1, 256, 256, 256, 256),
    ]
    im_c2s = [
        crop(im_c2, 0, 0, 256, 256),
        crop(im_c2, 0, 256, 256, 256),
        crop(im_c2, 256, 0, 256, 256),
        crop(im_c2, 256, 256, 256, 256),
    ]

    # Convert to tensors (NumPy arrays) and stack channels
    ims = [np.stack((to_numpy(x), to_numpy(y))) for x, y in zip(im_c1s, im_c2s)]

    for im in ims:
        im = normalize(im, mean, std)

    # Final stacking of all patches
    ims = np.stack(ims)

    return ims


def predict(model, image):
    """Make a prediction."""

    input_name = "input.1"
    output = model.run(None, {input_name: image})
    return output[0]


def black_and_white(pred_image):
    """Convert values to black and white."""

    bw_image = np.where(pred_image == 1, 255, 0).astype(np.uint8)

    # Convert into Image and returns it.
    bw_image_pil = Image.fromarray(bw_image)
    return bw_image_pil


def save(output, output_path):
    """Save the predicted picture in a black and white tif file."""

    # Convert probablities into classes.
    output = np.argmax(output, axis=1)

    # Reassemble the prediction picture.
    top_left, top_right, bottom_left, bottom_right = (
        output[0],
        output[1],
        output[2],
        output[3],
    )
    top_row = np.concatenate((top_left, top_right), axis=1)
    bottom_row = np.concatenate((bottom_left, bottom_right), axis=1)
    reconstructed_image = np.concatenate((top_row, bottom_row), axis=0)

    os.makedirs(output_path.parent, exist_ok=True)

    # Save it in predictions folder.
    pred_image_pil = black_and_white(reconstructed_image)
    pred_image_pil.save(output_path)
    print(f"image saved at: {output_path}")


def predict_image(model, input_path, output_path):
    """Prepare input, predict and save to output."""
    input_inference = prepare_input_inference(input_path)
    image = process_inference(input_inference)
    # Run prediction.
    output = predict(model, image)
    # Save output image.
    save(output, output_path)


def run():
    """Perform an inference and save the results."""

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Model path")
    parser.add_argument("input_path", type=str, help="Input path")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=Path("predictions"),
        type=Path,
        help="Output directory path",
    )
    parser.add_argument(
        "-s",
        "--stac-output",
        action="store_true",
        help="Generate STAC catalog as output",
    )

    args = parser.parse_args()

    # Load model.
    model_path = Path(args.model_path)
    model = load_model(model_path)

    input_path = Path(args.input_path)
    if input_path.is_file():
        input_images = [args.output_dir / f"prediction-{input_path.name}"]
    elif input_path.is_dir():
        input_images = [
            input_path / fpath
            for fpath in glob.glob("**/*.tif", root_dir=input_path, recursive=True)
        ]

    if args.stac_output:
        print(f"Generating STAC catalog at: {args.output_dir}")
        curr_datetime = datetime.datetime.now(tz=datetime.timezone.utc)
        stac_catalog = pystac.Catalog(
            id=f"flood-inference_{curr_datetime.strftime('%y-%m-%d')}_{str(uuid.uuid4())[:8]}",
            description="Flood model inference.",
        )
        stac_item = pystac.Item(
            id="data",
            geometry=None,
            bbox=None,
            datetime=curr_datetime,
            properties={},
        )

        for image_path in input_images:
            mime_type, _ = mimetypes.guess_type(image_path)
            media_type = mime_type if mime_type else "application/octet-stream"
            asset_path: Path = (
                args.output_dir / stac_item.id / f"prediction-{image_path.name}"
            )
            stac_item.add_asset(
                key=asset_path.name,
                asset=pystac.Asset(href=asset_path.name, media_type=media_type),
            )
            predict_image(model, image_path, asset_path)

        stac_catalog.add_item(stac_item)
        stac_catalog.normalize_and_save(
            str(args.output_dir),
            catalog_type=pystac.CatalogType.SELF_CONTAINED,
        )
        print("STAC catalog generated")
    else:
        for image_path in input_images:
            predict_image(
                model, image_path, args.output_dir / f"prediction-{image_path.name}"
            )


if __name__ == "__main__":
    run()
