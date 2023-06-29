import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn


def main():
    preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
    model = smp.FPN('resnet34', in_channels=1)
    # model = smp.DeepLabV3Plus('mit_b0', in_channels=3)
    model = smp.FPN('mit_b3', in_channels=3)
    mask = model(torch.ones([1, 3, 64, 64]))


if __name__ == '__main__':
    main()
