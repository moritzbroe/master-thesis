from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.utils.data import random_split, DataLoader
from torchvision import datasets


# get the standard dataloaders, i.e. train and test loader

# Mean and standard deviation for each dataset
STD = {
    'stl10': (0.229, 0.224, 0.225),
    'cifar10': (0.2470, 0.2435, 0.2616),
    'mnist': (0.3081,),
    'shapes': (1.0,),
    'imagenet': (0.229, 0.224, 0.225)
}
MEAN = {
    'stl10': (0.485, 0.456, 0.406),
    'cifar10': (0.4914, 0.4822, 0.4465),
    'mnist': (0.1307,),
    'shapes': (0.0,),
    'imagenet': (0.485, 0.456, 0.406)
}


def get_transform(dataset_name, normalize=True, resolution=None):
    if dataset_name not in MEAN:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    mean = MEAN[dataset_name]
    std = STD[dataset_name]

    transform_list = []
    if dataset_name in ['cifar10', 'mnist', 'stl10']:
        if resolution is not None:
            transform_list.append(transforms.Resize(resolution))
        transform_list.append(transforms.ToTensor())
        if normalize: 
            transform_list.append(transforms.Normalize(mean, std))
    elif dataset_name == 'imagenet':
        transform_list.append(transforms.Resize(256))
        transform_list.append(transforms.CenterCrop(224))
        if resolution is not None:
            transform_list.append(transforms.Resize(resolution))
        transform_list.append(transforms.ToTensor())
        if normalize:
            transform_list.append(transforms.Normalize(mean, std))
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return transforms.Compose(transform_list)


def get_train_test_loader(dataset_name, batch_size=128, num_workers=8, dataset_dir='./data', normalize=True, resolution=None):
    # For the final evaluation, we want to use the full training set and the test set.
    transform = get_transform(dataset_name, normalize, resolution)
    if dataset_name == 'cifar10':
        full_train_dataset = datasets.CIFAR10(root=dataset_dir, download=True, transform=transform, train=True)
        test_dataset = datasets.CIFAR10(root=dataset_dir, download=True, transform=transform, train=False)
    elif dataset_name == 'mnist':
        full_train_dataset = datasets.MNIST(root=dataset_dir, download=True, transform=transform, train=True)
        test_dataset = datasets.MNIST(root=dataset_dir, download=True, transform=transform, train=False)
    elif dataset_name == 'stl10':
        full_train_dataset = datasets.STL10(root=dataset_dir, split='train', download=True, transform=transform)
        test_dataset = datasets.STL10(root=dataset_dir, split='test', download=True, transform=transform)
    elif dataset_name == 'imagenet':
        full_train_dataset = datasets.ImageFolder(root=f"{dataset_dir}/imagenet/train", transform=transform)
        test_dataset = datasets.ImageFolder(root=f"{dataset_dir}/imagenet/val", transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    return train_loader, test_loader
    

