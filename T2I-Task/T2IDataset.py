from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import pandas as pd


class T2I_Dataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.captions = pd.read_csv(os.path.join(data_dir, "captions.csv"))
        self.transforms = transforms

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_filename = str(self.captions.iloc[idx, 0]) + ".jpg"
        img_path = os.path.join(self.data_dir, "Images", img_filename)
        image = read_image(img_path)
        caption = self.captions.iloc[idx, 1]

        # if self.transforms:
        #     image = self.transforms(image)

        return caption, image


if __name__ == "__main__":
    print(T2I_Dataset("Test Data").__getitem__(1))
