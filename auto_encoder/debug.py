import os
import glob
from tqdm import tqdm

import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))

        return x


class AEDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        # elf.targets = targets
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


def main():
    print('training an auto encoder')
    im_paths = glob.glob(os.path.join(r'C:\data\auto_encoder_samples', '*.jpg'), recursive=True)

    # create data transforms
    load_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    ds = AEDataset(images_filepaths=im_paths, transform=load_transforms)
    loader = DataLoader(dataset=ds, batch_size=4, shuffle=True)

    # Model Initialization
    # model = AE()
    model = ConvAutoencoder()

    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8)

    epochs = 10
    outputs = []
    losses = []
    for epoch in tqdm(range(epochs)):
        for (image, _) in loader:
            # Reshaping the image to (-1, 784)
            # image = image.reshape(-1, 28 * 28)

            # Output of Autoencoder
            reconstructed = model(image)

            # Calculating the loss function
            loss = loss_function(reconstructed, image)
            # print(loss)

            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            losses.append(loss.detach().numpy())
        outputs.append((epochs, image, reconstructed))

    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plotting the last 100 values
    plt.plot(losses[-100:])
    plt.show()


if __name__ == '__main__':
    main()
