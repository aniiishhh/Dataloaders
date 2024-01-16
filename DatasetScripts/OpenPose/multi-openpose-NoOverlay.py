"""
First download model through "model.sh"
Execute using:
    python multi-openpose-NoOverlay.py --i Input 
"""

import cv2
import numpy as np
import argparse
import os
from utils import *

parser = argparse.ArgumentParser(description="Openpose without overlay")
parser.add_argument("--i", help="Path to the directory containing input images.")
parser.add_argument(
    "--o",
    default="./Output_NoOverlay",
    help="Path to the directory to store output images.",
)
parser.add_argument(
    "--thr", default=0.1, type=float, help="Threshold value for pose parts heat map"
)
parser.add_argument(
    "--width", default=368, type=int, help="Resize input to a specific width."
)
parser.add_argument(
    "--height", default=368, type=int, help="Resize input to a specific height."
)
parser.add_argument("--scale", default=0.003922, type=float, help="Scale for blob.")

args = parser.parse_args()


inWidth = args.width
inHeight = args.height
inScale = args.scale

input_dir = args.i
output_dir = args.o
os.makedirs(output_dir, exist_ok=True)

protoFile = os.path.join("model", "pose_deploy_linevec.prototxt")
weightsFile = os.path.join("model", "pose_iter_440000.caffemodel")
nPoints = 18


keypointsMapping = [
    "Nose",
    "Neck",
    "R-Sho",
    "R-Elb",
    "R-Wr",
    "L-Sho",
    "L-Elb",
    "L-Wr",
    "R-Hip",
    "R-Knee",
    "R-Ank",
    "L-Hip",
    "L-Knee",
    "L-Ank",
    "R-Eye",
    "L-Eye",
    "R-Ear",
    "L-Ear",
]

POSE_PAIRS = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
    [2, 17],
    [5, 16],
]

mapIdx = [
    [31, 32],
    [39, 40],
    [33, 34],
    [35, 36],
    [41, 42],
    [43, 44],
    [19, 20],
    [21, 22],
    [23, 24],
    [25, 26],
    [27, 28],
    [29, 30],
    [47, 48],
    [49, 50],
    [53, 54],
    [51, 52],
    [55, 56],
    [37, 38],
    [45, 46],
]

colors = [
    [0, 100, 255],
    [0, 100, 255],
    [0, 255, 255],
    [0, 100, 255],
    [0, 255, 255],
    [0, 100, 255],
    [0, 255, 0],
    [255, 200, 100],
    [255, 0, 255],
    [0, 255, 0],
    [255, 200, 100],
    [255, 0, 255],
    [0, 0, 255],
    [255, 0, 0],
    [200, 200, 0],
    [255, 0, 0],
    [200, 200, 0],
    [0, 0, 0],
]


image_files = [
    f for f in os.listdir(input_dir) if f.endswith((".jpg", ".png", ".jpeg"))
]

for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    image1 = cv2.imread(image_path)

    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    inpBlob = cv2.dnn.blobFromImage(
        image1, inScale, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False
    )

    net.setInput(inpBlob)
    output = net.forward()

    detected_keypoints = []
    keypoints_list = np.zeros((0, 3))
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0, part, :, :]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        keypoints = getKeypoints(probMap, threshold)

        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)

    frameClone = image1.copy()
    valid_pairs, invalid_pairs = getValidPairs(
        output, mapIdx, POSE_PAIRS, image1, detected_keypoints
    )
    personwiseKeypoints = getPersonwiseKeypoints(
        valid_pairs, invalid_pairs, mapIdx, POSE_PAIRS, keypoints_list
    )

    white_background = np.ones_like(frameClone) * 255

    stick_figure = np.zeros_like(frameClone)

    for i in range(len(detected_keypoints)):
        for j in range(len(detected_keypoints[i])):
            keypoint = detected_keypoints[i][j]
            cv2.circle(stick_figure, keypoint[0:2], 5, colors[i], -1, cv2.LINE_AA)

    for i in range(17):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(
                stick_figure,
                (B[0], A[0]),
                (B[1], A[1]),
                colors[i],
                3,
                cv2.LINE_AA,
            )

    mask = cv2.bitwise_not(stick_figure)

    result = cv2.subtract(white_background, mask)

    output_path = os.path.join(output_dir, f"openpose_{image_file}")
    cv2.imwrite(output_path, result)

    print(f"Successfully written openpose_{image_file}")


cv2.destroyAllWindows()
print("-" * 150)
