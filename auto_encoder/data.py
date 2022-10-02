from torch.utils.data import Dataset, DataLoader
import torch
import cv2


class PlantPathologyDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            # image = self.transform(image=image)['image']
            image = self.transform(image)

        # label = torch.tensor(self.targets[idx]).long()
        # label = torch.tensor(self.targets[idx]).float()
        return image, image

