from torch.utils.data import Dataset, DataLoader


class PaddyDataset(Dataset):
    def __init__(self, images_filepaths, targets, transform=None):
        self.images_filepaths = images_filepaths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv.imread(image_filepath)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        label = torch.tensor(self.targets[idx]).long()
        return image, label
