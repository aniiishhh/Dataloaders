"""
Execute using:
    python final-scribble.py --i <Input Folder> --o <Output Folder> --cuda
"""

import argparse
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
import numpy as np
import os
import torch


def process_images(input_folder, output_folder, use_cuda=False):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators").to(device)

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            image = load_image(image_path)

            image_np = np.array(image)

            image_tensor = torch.tensor(image_np).to(device)

            hed_result = hed(image_tensor, scribble=True)

            filename = filename.replace(".", "_scribble.")
            output_path = os.path.join(output_folder, filename)
            hed_result.save(output_path)
            print(f"Successfully written {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process Scribble on images")
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
