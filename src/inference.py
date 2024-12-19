""" This file can be used to perform an inference on the trained flood detection model."""

import argparse
import os
import numpy as np

import onnxruntime as ort
from PIL import Image
import rasterio


def load_model(path):
    """Load the model in evaluation mode."""

    ort_session = ort.InferenceSession(path)
    return ort_session


def get_arr_flood(fname):
    """open image with rasterio."""
    return rasterio.open(fname).read()


def prepare_input_inference(data):
    """prepare input."""
    arr_x = np.nan_to_num(get_arr_flood(data))
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


def save(output, save_path="predictions/"):
    """Save the predicted picture in a black and white tif file."""

    if not os.path.exists(save_path):
        os.makedirs(save_path)

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

    # Save it in predictions folder.
    pred_image_pil = black_and_white(reconstructed_image)
    pred_image_pil.save(f"{save_path}/prediction.tif")
    print(f"image saved at {save_path}prediction.tif")


def run():
    """Perfom an inference and save the results.
    Outputs: predictions/prediction.tif
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("param1", type=str, help="model_path")
    parser.add_argument("param2", type=str, help="input_path")

    args = parser.parse_args()

    # Load model.
    model = load_model(args.param1)

    # Prepare Input.
    input_inference = prepare_input_inference(args.param2)
    image = process_inference(input_inference)

    # Run prediction.
    output = predict(model, image)

    # Metrics and save.
    save(output, save_path="predictions/")


if __name__ == "__main__":
    run()
