"""
Execute using:
    python final-normal.py --i <Input Folder> --o <Output Folder> --cuda
"""

import argparse
from PIL import Image
from transformers import pipeline
import numpy as np
import cv2
import os
import torch
from diffusers.utils import load_image


def process_images(input_folder, output_folder, use_cuda):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            image = load_image(image_path).convert("RGB")

            image_tensor = torch.tensor(np.array(image)).to(device)

            depth_result = (
                depth_estimator(image_tensor)["predicted_depth"][0].cpu().numpy()
            )

            image_depth = depth_result.copy()
            image_depth -= np.min(image_depth)
            image_depth /= np.max(image_depth)

            bg_threshold = 0.4

            x = cv2.Sobel(depth_result, cv2.CV_32F, 1, 0, ksize=3)
            x[image_depth < bg_threshold] = 0

            y = cv2.Sobel(depth_result, cv2.CV_32F, 0, 1, ksize=3)
            y[image_depth < bg_threshold] = 0

            z = np.ones_like(x) * np.pi * 2.0

            final_result = np.stack([x, y, z], axis=2)
            final_result /= np.sum(final_result**2.0, axis=2, keepdims=True) ** 0.5
            final_result = (final_result * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
            final_image = Image.fromarray(final_result)

            output_filename = filename.replace(".", "_normal.")
            output_path = os.path.join(output_folder, output_filename)
            final_image.save(output_path)
            print(f"Successfully written {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process NORMAL on images")
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
