from torch.utils.data import Dataset
from code_utils import read_image_with_any_extension
import os
import pandas as pd


class TextGuidedEditing_Dataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.instructions = pd.read_csv(os.path.join(data_dir, "instructions.csv"))
        self.transforms = transforms

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        img_filename = str(self.instructions.iloc[idx, 0])
        input_image_path = os.path.join(self.data_dir, "Input-Images", img_filename)
        target_image_path = os.path.join(self.data_dir, "Target-Images", img_filename)

        input_image = read_image_with_any_extension(input_image_path)
        target_image = read_image_with_any_extension(target_image_path)

        # if self.transforms:
        #     input_image = self.transforms(input_image)
        #     target_image = self.transforms(target_image)

        instruction = self.instructions.iloc[idx, 1]

        return (input_image, instruction, target_image)


if __name__ == "__main__":
    print(TextGuidedEditing_Dataset("Test Data").__getitem__(0))
