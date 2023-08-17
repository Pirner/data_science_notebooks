import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from modells import CNN
from trainer import MNISTTrainer


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )

    loaders = {
        'train': torch.utils.data.DataLoader(
            train_data,
            batch_size=100,
            shuffle=True,
            num_workers=1),
        'test': torch.utils.data.DataLoader(
            test_data,
            batch_size=100,
            shuffle=True,
            num_workers=1),
    }

    cnn = CNN()
    # print(cnn)
    # training code
    # loss_func = nn.CrossEntropyLoss()
    trainer = MNISTTrainer()
    trainer.train(cnn, 10, loaders)


if __name__ == '__main__':
    main()
