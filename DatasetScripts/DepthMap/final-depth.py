"""
Execute using:
    python final-depth.py --i <Input Folder> --o <Output Folder> --cuda
"""

import argparse
from diffusers import pipeline, utils
import os
import torch
import numpy as np


def process_images(input_folder, output_folder, use_cuda=False):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    depth_estimator = pipeline("depth-estimation").to(device)

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            image = utils.load_image(image_path)

            image_np = np.array(image)

            image_tensor = torch.tensor(image_np).to(device)

            depth_map = depth_estimator(image_tensor)["depth"]

            filename = filename.replace(".", "_depth.")
            output_path = os.path.join(output_folder, filename)
            depth_map.save(output_path)
            print(f"Successfully written {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process depth estimation on images"
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
