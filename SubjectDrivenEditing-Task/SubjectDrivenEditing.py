from torch.utils.data import Dataset
from code_utils import read_image_with_any_extension, num_images_in_directory
import os
import pandas as pd


class SubjectDrivenEditing_Dataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        image_dir = os.path.join(self.data_dir, "Input-Images")
        return num_images_in_directory(image_dir)

    def __getitem__(self, idx):
        img_filename = str(idx)
        input_image_path = os.path.join(self.data_dir, "Input-Images", img_filename)
        token_image_path = os.path.join(self.data_dir, "Token-Images", img_filename)
        target_image_path = os.path.join(self.data_dir, "Target-Images", img_filename)

        input_image = read_image_with_any_extension(input_image_path)
        token_image = read_image_with_any_extension(token_image_path)
        target_image = read_image_with_any_extension(target_image_path)

        # if self.transforms:
        #     input_image = self.transforms(input_image)
        #     token_image = self.transforms(token_image)
        #     target_image = self.transforms(target_image)

        return (input_image, token_image, target_image)


if __name__ == "__main__":
    print(SubjectDrivenEditing_Dataset("Test Data").__len__())
    print(SubjectDrivenEditing_Dataset("Test Data").__getitem__(1))
