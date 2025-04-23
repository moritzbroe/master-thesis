import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import matplotlib.pyplot as plt


# this defines the dataloaders for getting the augmented pairs of images for vicreg. as the cpu was often the bottleneck, it uses the faster albumentations library for augmentations.

# Normalization parameters for various datasets.
NORMALIZATION_PARAMS = {
    'stl10': ((0.4408, 0.4279, 0.3867), (0.2682, 0.2610, 0.2686)),
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'mnist': ((0.1307,), (0.3081,)),
    'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}

# simple dataset which returns pairs of augmented images, but don't define transformations yet
class SSLUnlabeledDataset(Dataset):
    def __init__(self, dataset_name, dataset_dir='./data'):
        self.dataset_name = dataset_name.lower()
        self.transform = None
        if self.dataset_name == 'stl10':
            self.dataset = datasets.STL10(root=dataset_dir, split='unlabeled', download=True, transform=None)
        elif self.dataset_name == 'cifar10':
            self.dataset = datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=None)
        elif self.dataset_name == 'mnist':
            self.dataset = datasets.MNIST(root=dataset_dir, train=True, download=True, transform=None)
        elif self.dataset_name == 'imagenet':
            self.dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, "imagenet", "train"), transform=None)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported for SSL unlabeled data.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        # Return two augmented versions of the same image.
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2

# set the transformations for the dataloader. code uses this weird structure so that the dataset doesn't have to be reloaded for each layer's training and instead just the transformations are changed.
def set_transformations(dataloader, color_jitter_strength, min_size, flip, res):
    dataset_name = dataloader.dataset.dataset_name 
    mean, std = NORMALIZATION_PARAMS[dataset_name]

    # Build the albumentations transformation pipeline.
    transform_list = []
    transform_list.append(A.RandomResizedCrop(size=(res, res), scale=(min_size, 1.0), p=1.0))
    if flip:
        flip_p = 0.5 if flip is True else flip
        transform_list.append(A.HorizontalFlip(p=flip_p))
    transform_list.append(A.ColorJitter(
        brightness=color_jitter_strength,
        contrast=color_jitter_strength,
        saturation=color_jitter_strength,
        hue=color_jitter_strength / 4,
        p=1.0
    ))
    transform_list.append(A.Normalize(mean=list(mean), std=list(std)))
    transform_list.append(ToTensorV2())

    albumentations_transform = A.Compose(transform_list)

    # Wrap the albumentations transform to work with PIL images.
    def transform_fn(img):
        img_np = np.array(img)
        # If the image is grayscale, convert it to 3 channels.
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        result = albumentations_transform(image=img_np)
        return result['image']

    dataloader.dataset.transform = transform_fn
    return dataloader

def get_basic_ssl_loader(dataset_name, num_workers=10, batch_size=256, dataset_dir='./data'):
    dataset = SSLUnlabeledDataset(dataset_name, dataset_dir=dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            drop_last=True, persistent_workers=False) # DO NOT EVER USE PERSISTENT WORKERS HERE (it leads to the augmentations not being changed sometimes which cost me days to find out)
    return dataloader    


# the rest of the code is for visualizing the augmented images
def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def show_samples(dataloader, dataset_name, num_samples=4):
    dataset_name = dataset_name.lower()
    if dataset_name not in NORMALIZATION_PARAMS:
        raise ValueError(f"Normalization parameters for {dataset_name} are not defined.")
    mean, std = NORMALIZATION_PARAMS[dataset_name]

    # Get one batch from the dataloader.
    batch = next(iter(dataloader))
    imgs1, imgs2 = batch
    imgs1 = imgs1[:num_samples]
    imgs2 = imgs2[:num_samples]

    # Unnormalize images for display.
    imgs1 = torch.stack([unnormalize(img.clone(), mean, std) for img in imgs1])
    imgs2 = torch.stack([unnormalize(img.clone(), mean, std) for img in imgs2])

    # For MNIST, images are single-channel so squeeze the channel dimension.
    if imgs1.shape[1] == 1:
        imgs1 = imgs1.squeeze(1)
        imgs2 = imgs2.squeeze(1)
        is_gray = True
    else:
        imgs1 = imgs1.permute(0, 2, 3, 1)
        imgs2 = imgs2.permute(0, 2, 3, 1)
        is_gray = False

    # Plot the augmented image pairs.
    fig, axes = plt.subplots(2, num_samples)
    for i in range(num_samples):
        if is_gray:
            axes[0, i].imshow(np.clip(imgs1[i].numpy(), 0, 1), cmap='gray')
            axes[1, i].imshow(np.clip(imgs2[i].numpy(), 0, 1), cmap='gray')
        else:
            axes[0, i].imshow(np.clip(imgs1[i].numpy(), 0, 1))
            axes[1, i].imshow(np.clip(imgs2[i].numpy(), 0, 1))
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    dataset_name = 'stl10'
    dataloader = get_basic_ssl_loader(dataset_name, num_workers=5, batch_size=256)
    # dataloader = set_transformations(dataloader, color_jitter_strength=0.684, min_size=0.9708, flip=0.022, res=96)
    # show_samples(dataloader, dataset_name, num_samples=3)
    dataloader = set_transformations(dataloader, color_jitter_strength=0.21, min_size=0.21, flip=0.17, res=96)
    show_samples(dataloader, dataset_name, num_samples=3)
