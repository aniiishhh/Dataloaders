"""
Execute using:
python main.py --i Input --o Output --m <model_type>
    # model_type = "DPT_Large"    MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
"""

import os
import cv2
import torch
import ssl
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="Depth Map Generation Script.")
parser.add_argument("--i", type=str, required=True, help="Path to the input directory.")
parser.add_argument(
    "--o", type=str, default="./Output", help="Path to the output directory."
)
parser.add_argument(
    "--m",
    choices=[
        "DPT_Large",
        "DPT_Hybrid",
        "MiDaS_small",
    ],
    default="MiDaS_small",
    help="Choose a model. Default is MiDaS_small.",
)
args = parser.parse_args()

input_dir = args.i
output_dir = args.o
os.makedirs(output_dir, exist_ok=True)
model_type = args.m

if "TORCH_HOME" not in os.environ:
    os.environ["TORCH_HOME"] = "../Models"


try:
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
except:
    print(f"Disabling SSL Verification and trying again.")

    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
    except Exception as e:
        print(f"Unable to load the model. Error: {e}")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


image_files = [
    f for f in os.listdir(input_dir) if f.endswith((".jpg", ".png", ".jpeg"))
]
for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    depth_map_normalized = (output - output.min()) / (output.max() - output.min()) * 255
    depth_map_uint8 = depth_map_normalized.astype(np.uint8)

    cv2.imwrite(os.path.join("./", output_dir, f"depth_{image_file}"), depth_map_uint8)

    print(f"Successfully written depth_{image_file}")

print("-" * 150)


if "TORCH_HOME" in os.environ:
    os.environ.pop("TORCH_HOME")
