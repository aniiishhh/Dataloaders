from torchvision.io import read_image
import os


def read_image_with_any_extension(img_path):
    image_extensions = [".jpg", ".jpeg", ".png"]
    for ext in image_extensions:
        img_path_with_ext = img_path + ext
        if os.path.exists(img_path_with_ext):
            image = read_image(img_path_with_ext)
            return image

    raise FileNotFoundError(f"Could not find image at path: {img_path}")
