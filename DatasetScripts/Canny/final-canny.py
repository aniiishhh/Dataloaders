"""
Execute using:
    python final-canny.py --i <Input Folder> --o <Output Folder> --cuda
"""

import argparse
import cv2
import os
import numpy as np
from PIL import Image
from diffusers.utils import load_image
import torch


def process_images(input_folder, output_folder, use_cuda=False):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            image = load_image(image_path)
            image_np = np.array(image)

            low_threshold = 100
            high_threshold = 200

            edges = cv2.Canny(
                image_np,
                low_threshold,
                high_threshold,
                use_cuda=(device == True),
            )

            edges_image = Image.fromarray(edges)

            filename = filename.replace(".", "_canny.")
            output_path = os.path.join(output_folder, filename)
            edges_image.save(output_path)
            print(f"Successfully written {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process Canny edge detection on images"
    )
    parser.add_argument(
        "--i", dest="input_folder", required=True, help="Input folder containing images"
    )
    parser.add_argument(
        "--o",
        dest="output_folder",
        default="Output",
        help="Output folder for processed images",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()

    process_images(args.input_folder, args.output_folder, use_cuda=args.cuda)
    print("-" * 150)
