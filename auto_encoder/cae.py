import os
import glob
from tqdm import tqdm

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch

from auto_encoder.data import PlantPathologyDataset


# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # encoder layers #
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        # decoder layers #
        # a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        # encode #
        # add hidden layers with relu activation function
        # and max pooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation

        # decode #
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))

        return x


def main():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Create training and test dataloaders
    data_root_dir = r'C:\kaggle\plant-pathology-2021-fgvc8\train_images'
    im_paths = glob.glob(os.path.join(data_root_dir, '*.jpg'), recursive=True)

    num_workers = 0
    # how many samples per batch to load
    batch_size = 20

    # prepare data loaders
    train_dataset = PlantPathologyDataset(
        images_filepaths=im_paths,
        transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    # initialize the NN
    model = ConvAutoencoder()

    # specify loss function
    criterion = nn.BCELoss()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # number of epochs to train the model
    n_epochs = 100

    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        for data in tqdm(train_loader):
            # _ stands in for labels, here
            # no need to flatten images
            images, _ = data
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # calculate the loss
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * images.size(0)

        # print avg training statistics
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


if __name__ == '__main__':
    main()
