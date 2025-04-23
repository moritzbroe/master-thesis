import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# create mnist datasets, in particular:
# - standard labeled train + test sets
# - unlabeled train set
# - ssl dataset with unlabeled, augmented image pairs


# precomputed pixelwise mean for pixelwise centering
PIXELWISE_MEAN = torch.tensor([[[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 8.2353e-06, 3.0719e-05, 1.4118e-05,
          5.8824e-07, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0458e-06,
          3.5948e-06, 3.6405e-05, 9.5229e-05, 1.7144e-04, 2.5137e-04,
          4.7111e-04, 6.3033e-04, 6.8307e-04, 6.9582e-04, 7.4242e-04,
          6.8294e-04, 7.3307e-04, 6.0255e-04, 3.9261e-04, 2.7935e-04,
          2.1105e-04, 8.3791e-05, 3.9542e-05, 1.3856e-05, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 4.1830e-06, 2.7451e-06, 2.7255e-05,
          2.1503e-05, 1.8471e-04, 5.4275e-04, 1.0360e-03, 1.9867e-03,
          3.3992e-03, 5.0592e-03, 7.3347e-03, 9.9214e-03, 1.2555e-02,
          1.4218e-02, 1.4596e-02, 1.3304e-02, 1.0992e-02, 8.0172e-03,
          4.7142e-03, 2.4841e-03, 1.1614e-03, 3.6856e-04, 1.3810e-04,
          3.3856e-05, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 1.2680e-05, 2.2941e-05, 4.7124e-05,
          2.7359e-04, 8.3170e-04, 2.1416e-03, 4.5271e-03, 8.6898e-03,
          1.4273e-02, 2.1325e-02, 2.9047e-02, 3.8026e-02, 4.6600e-02,
          5.1911e-02, 5.1469e-02, 4.6328e-02, 3.7426e-02, 2.6914e-02,
          1.6446e-02, 8.9201e-03, 4.1609e-03, 1.6199e-03, 6.3562e-04,
          1.0889e-04, 1.0980e-05, 0.0000e+00],
         [0.0000e+00, 2.4837e-06, 2.0458e-05, 5.6275e-05, 3.1719e-04,
          1.6087e-03, 4.0911e-03, 9.4874e-03, 1.8728e-02, 3.2919e-02,
          5.2197e-02, 7.6382e-02, 1.0603e-01, 1.3809e-01, 1.6408e-01,
          1.7746e-01, 1.7388e-01, 1.5350e-01, 1.2286e-01, 8.9898e-02,
          5.8161e-02, 3.3966e-02, 1.7835e-02, 8.3805e-03, 3.3839e-03,
          8.1471e-04, 1.1627e-04, 7.9739e-06],
         [0.0000e+00, 0.0000e+00, 4.0458e-05, 2.4843e-04, 1.5508e-03,
          5.7388e-03, 1.4070e-02, 2.8344e-02, 5.1048e-02, 8.3208e-02,
          1.2353e-01, 1.7336e-01, 2.3100e-01, 2.8943e-01, 3.3374e-01,
          3.5529e-01, 3.4872e-01, 3.1444e-01, 2.5860e-01, 1.9527e-01,
          1.3471e-01, 8.4353e-02, 4.8589e-02, 2.6036e-02, 1.1736e-02,
          3.3085e-03, 5.5542e-04, 1.8431e-05],
         [0.0000e+00, 7.1895e-07, 1.0784e-04, 8.6837e-04, 4.3903e-03,
          1.2902e-02, 2.9163e-02, 5.5900e-02, 9.4658e-02, 1.4623e-01,
          2.0830e-01, 2.7809e-01, 3.5011e-01, 4.1660e-01, 4.6473e-01,
          4.8666e-01, 4.7730e-01, 4.3935e-01, 3.7689e-01, 2.9477e-01,
          2.1191e-01, 1.3848e-01, 8.2321e-02, 4.4872e-02, 2.1128e-02,
          7.2768e-03, 1.4616e-03, 1.1882e-04],
         [3.0719e-06, 7.7386e-05, 4.2222e-04, 2.3359e-03, 9.0682e-03,
          2.3250e-02, 4.8769e-02, 8.8046e-02, 1.4233e-01, 2.1137e-01,
          2.8927e-01, 3.6903e-01, 4.3757e-01, 4.9020e-01, 5.2062e-01,
          5.3098e-01, 5.2290e-01, 4.9677e-01, 4.4528e-01, 3.6762e-01,
          2.7364e-01, 1.8366e-01, 1.1037e-01, 5.9612e-02, 2.7583e-02,
          1.0130e-02, 2.0122e-03, 1.2503e-04],
         [1.5948e-05, 1.9928e-04, 1.2916e-03, 5.1661e-03, 1.4699e-02,
          3.3268e-02, 6.5954e-02, 1.1607e-01, 1.8404e-01, 2.6709e-01,
          3.5408e-01, 4.2539e-01, 4.6837e-01, 4.8390e-01, 4.8250e-01,
          4.7977e-01, 4.8129e-01, 4.7900e-01, 4.5593e-01, 3.9455e-01,
          3.0321e-01, 2.0744e-01, 1.2475e-01, 6.4148e-02, 2.8179e-02,
          1.0222e-02, 1.8886e-03, 1.0660e-04],
         [1.9085e-05, 3.1163e-04, 2.0079e-03, 6.7427e-03, 1.7435e-02,
          3.8457e-02, 7.6574e-02, 1.3531e-01, 2.1451e-01, 3.0642e-01,
          3.8839e-01, 4.3373e-01, 4.3446e-01, 4.1001e-01, 3.8842e-01,
          3.8976e-01, 4.0915e-01, 4.3483e-01, 4.3560e-01, 3.8881e-01,
          3.0212e-01, 2.0669e-01, 1.2333e-01, 6.0364e-02, 2.3377e-02,
          7.4430e-03, 1.3797e-03, 1.1065e-04],
         [2.6144e-05, 3.8654e-04, 2.0511e-03, 6.5670e-03, 1.6820e-02,
          3.8731e-02, 8.0546e-02, 1.4584e-01, 2.3418e-01, 3.2705e-01,
          3.9304e-01, 4.0458e-01, 3.6820e-01, 3.2531e-01, 3.1146e-01,
          3.3147e-01, 3.6811e-01, 4.0952e-01, 4.1679e-01, 3.6950e-01,
          2.8176e-01, 1.8948e-01, 1.1168e-01, 5.2934e-02, 1.8041e-02,
          4.5102e-03, 8.6993e-04, 7.3464e-05],
         [2.3268e-05, 2.9961e-04, 1.6505e-03, 5.0210e-03, 1.4298e-02,
          3.7466e-02, 8.2875e-02, 1.5629e-01, 2.5133e-01, 3.4200e-01,
          3.8737e-01, 3.7203e-01, 3.2033e-01, 2.8762e-01, 2.9861e-01,
          3.3500e-01, 3.8154e-01, 4.2150e-01, 4.1376e-01, 3.4914e-01,
          2.5468e-01, 1.6665e-01, 9.9377e-02, 4.9334e-02, 1.5647e-02,
          2.3707e-03, 4.8824e-04, 3.2026e-05],
         [1.4902e-05, 1.7131e-04, 9.4915e-04, 3.4463e-03, 1.2035e-02,
          3.7676e-02, 8.9325e-02, 1.7088e-01, 2.7048e-01, 3.5395e-01,
          3.8390e-01, 3.5719e-01, 3.1258e-01, 3.1125e-01, 3.5035e-01,
          3.9888e-01, 4.4329e-01, 4.6110e-01, 4.2128e-01, 3.3145e-01,
          2.2975e-01, 1.5005e-01, 9.3095e-02, 4.9851e-02, 1.6842e-02,
          1.7382e-03, 3.1288e-04, 3.9869e-05],
         [2.0915e-06, 7.1569e-05, 4.8085e-04, 2.3817e-03, 1.1442e-02,
          4.1703e-02, 1.0021e-01, 1.8660e-01, 2.8483e-01, 3.5950e-01,
          3.7995e-01, 3.5725e-01, 3.4066e-01, 3.8026e-01, 4.3675e-01,
          4.8616e-01, 5.0905e-01, 4.9647e-01, 4.2698e-01, 3.1827e-01,
          2.1657e-01, 1.4590e-01, 9.4011e-02, 5.3505e-02, 2.0011e-02,
          2.2929e-03, 3.2542e-04, 4.2614e-05],
         [7.3856e-06, 3.2353e-05, 2.1242e-04, 1.8282e-03, 1.2022e-02,
          4.8436e-02, 1.1187e-01, 1.9807e-01, 2.8922e-01, 3.5439e-01,
          3.7285e-01, 3.6584e-01, 3.8364e-01, 4.5263e-01, 5.1085e-01,
          5.4727e-01, 5.3765e-01, 5.0230e-01, 4.1959e-01, 3.1198e-01,
          2.1999e-01, 1.5272e-01, 1.0015e-01, 5.7319e-02, 2.2438e-02,
          3.2169e-03, 3.6255e-04, 8.6928e-06],
         [2.8758e-06, 1.5163e-05, 1.7686e-04, 1.9827e-03, 1.3963e-02,
          5.6503e-02, 1.2152e-01, 2.0082e-01, 2.8016e-01, 3.3620e-01,
          3.5836e-01, 3.6958e-01, 4.1195e-01, 4.8315e-01, 5.3212e-01,
          5.4553e-01, 5.1688e-01, 4.7622e-01, 3.9741e-01, 3.0737e-01,
          2.2856e-01, 1.6177e-01, 1.0580e-01, 5.8395e-02, 2.2856e-02,
          4.2336e-03, 5.8124e-04, 4.2484e-05],
         [2.6144e-06, 1.5556e-05, 2.9150e-04, 2.4430e-03, 1.7529e-02,
          6.4819e-02, 1.2767e-01, 1.9586e-01, 2.6005e-01, 3.0544e-01,
          3.2826e-01, 3.5078e-01, 3.9661e-01, 4.5434e-01, 4.9597e-01,
          4.9971e-01, 4.7473e-01, 4.3615e-01, 3.7401e-01, 3.0335e-01,
          2.3375e-01, 1.6486e-01, 1.0482e-01, 5.5855e-02, 2.2022e-02,
          5.0475e-03, 7.5098e-04, 4.8235e-05],
         [0.0000e+00, 2.8562e-05, 4.4000e-04, 3.7386e-03, 2.3342e-02,
          7.2645e-02, 1.3118e-01, 1.8893e-01, 2.3710e-01, 2.7144e-01,
          2.9287e-01, 3.1661e-01, 3.5132e-01, 3.9981e-01, 4.4163e-01,
          4.5269e-01, 4.4018e-01, 4.0990e-01, 3.6324e-01, 3.0299e-01,
          2.3266e-01, 1.6003e-01, 9.7815e-02, 5.0333e-02, 2.0222e-02,
          5.5556e-03, 8.2366e-04, 3.4837e-05],
         [7.4510e-06, 2.0850e-05, 7.0784e-04, 5.9614e-03, 2.9888e-02,
          8.0070e-02, 1.3691e-01, 1.8895e-01, 2.2854e-01, 2.5880e-01,
          2.8136e-01, 3.0033e-01, 3.2597e-01, 3.7306e-01, 4.1787e-01,
          4.4001e-01, 4.3666e-01, 4.1184e-01, 3.6713e-01, 3.0050e-01,
          2.2236e-01, 1.4809e-01, 8.8418e-02, 4.4637e-02, 1.7957e-02,
          5.0924e-03, 6.0366e-04, 4.7778e-05],
         [9.8039e-07, 4.7778e-05, 1.1228e-03, 8.0150e-03, 3.4210e-02,
          8.5233e-02, 1.4444e-01, 1.9992e-01, 2.4451e-01, 2.7950e-01,
          3.0541e-01, 3.2406e-01, 3.5281e-01, 3.9767e-01, 4.4096e-01,
          4.6197e-01, 4.5461e-01, 4.2130e-01, 3.6187e-01, 2.8190e-01,
          1.9922e-01, 1.2821e-01, 7.3619e-02, 3.5907e-02, 1.4275e-02,
          4.1902e-03, 5.7529e-04, 2.6797e-05],
         [0.0000e+00, 5.9869e-05, 1.3020e-03, 8.8458e-03, 3.3385e-02,
          8.1153e-02, 1.4543e-01, 2.1096e-01, 2.7008e-01, 3.1827e-01,
          3.5453e-01, 3.8387e-01, 4.1857e-01, 4.6015e-01, 4.9001e-01,
          4.9238e-01, 4.6359e-01, 4.0708e-01, 3.2893e-01, 2.4014e-01,
          1.6060e-01, 9.7820e-02, 5.3446e-02, 2.5637e-02, 1.0643e-02,
          2.9379e-03, 4.3797e-04, 6.6013e-06],
         [2.0915e-06, 4.9804e-05, 1.1146e-03, 7.2125e-03, 2.6010e-02,
          6.6276e-02, 1.2866e-01, 2.0224e-01, 2.7617e-01, 3.4216e-01,
          3.9602e-01, 4.4100e-01, 4.8076e-01, 5.1126e-01, 5.1704e-01,
          4.8968e-01, 4.3147e-01, 3.5085e-01, 2.6114e-01, 1.7881e-01,
          1.1199e-01, 6.3905e-02, 3.3378e-02, 1.6114e-02, 6.6784e-03,
          1.7405e-03, 2.3157e-04, 2.5490e-06],
         [2.0261e-06, 3.8562e-06, 7.3366e-04, 4.2665e-03, 1.5468e-02,
          4.2531e-02, 9.1728e-02, 1.5937e-01, 2.3909e-01, 3.1767e-01,
          3.8867e-01, 4.4353e-01, 4.8049e-01, 4.9138e-01, 4.7299e-01,
          4.2034e-01, 3.4395e-01, 2.5872e-01, 1.7778e-01, 1.1267e-01,
          6.6288e-02, 3.5927e-02, 1.8467e-02, 8.7256e-03, 3.2750e-03,
          7.4163e-04, 6.9935e-05, 4.7059e-06],
         [0.0000e+00, 0.0000e+00, 2.5163e-04, 1.6422e-03, 6.7949e-03,
          1.9666e-02, 4.7180e-02, 9.3883e-02, 1.5787e-01, 2.3095e-01,
          3.0302e-01, 3.6091e-01, 3.9100e-01, 3.8888e-01, 3.5526e-01,
          2.9641e-01, 2.2583e-01, 1.5762e-01, 9.9989e-02, 5.9764e-02,
          3.3426e-02, 1.7330e-02, 8.6473e-03, 3.8496e-03, 1.2171e-03,
          2.2732e-04, 3.7712e-05, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 6.2157e-05, 4.7922e-04, 2.1319e-03,
          6.2778e-03, 1.6484e-02, 3.5857e-02, 6.5991e-02, 1.0618e-01,
          1.4943e-01, 1.8449e-01, 2.0239e-01, 1.9985e-01, 1.7819e-01,
          1.4410e-01, 1.0761e-01, 7.4936e-02, 4.7506e-02, 2.8294e-02,
          1.5527e-02, 7.8164e-03, 3.7303e-03, 1.5672e-03, 3.9928e-04,
          8.6993e-05, 7.5817e-06, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 6.1438e-06, 6.9739e-05, 5.0111e-04,
          1.8608e-03, 5.5118e-03, 1.2474e-02, 2.4092e-02, 3.8544e-02,
          5.4390e-02, 6.5562e-02, 7.0980e-02, 6.9707e-02, 6.1991e-02,
          5.1465e-02, 4.1169e-02, 3.0554e-02, 2.0478e-02, 1.2401e-02,
          6.6327e-03, 3.2368e-03, 1.4533e-03, 5.4830e-04, 1.2261e-04,
          1.3987e-05, 6.7974e-06, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00, 2.4837e-06, 1.3902e-04,
          6.3954e-04, 2.1109e-03, 4.7033e-03, 9.0827e-03, 1.3770e-02,
          1.9020e-02, 2.3452e-02, 2.5274e-02, 2.4546e-02, 2.1845e-02,
          1.7454e-02, 1.3862e-02, 1.0164e-02, 6.6712e-03, 3.9553e-03,
          2.1188e-03, 9.3490e-04, 2.9516e-04, 6.3399e-05, 2.0261e-06,
          3.8562e-06, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 9.9346e-06,
          6.1111e-05, 1.6327e-04, 3.4980e-04, 5.0281e-04, 7.7150e-04,
          1.3177e-03, 1.6861e-03, 2.0627e-03, 2.3164e-03, 2.6982e-03,
          2.3218e-03, 1.8931e-03, 1.3471e-03, 7.8601e-04, 3.4850e-04,
          1.7895e-04, 7.5621e-05, 5.9281e-05, 7.8431e-06, 0.0000e+00,
          0.0000e+00, 0.0000e+00, 0.0000e+00]]])


STANDARD_MEAN = (0.1307,)
STANDARD_STD = (0.3081,)


class SubtractPixelwiseMean:
    """Subtracts the precomputed pixelwise mean from a tensor."""
    def __call__(self, tensor):
        return tensor - PIXELWISE_MEAN

def get_transform(normalization="standard"):
    """
    Returns the transform pipeline for MNIST based on the normalization argument.
    
    normalization:
      - "none": Only converts the image to a tensor.
      - "standard": Converts to tensor and applies standard normalization.
      - "pixelwise": Converts to tensor and subtracts the precomputed pixelwise mean.
    """
    transform_list = [transforms.ToTensor()]
    
    if normalization == "standard":
        transform_list.append(transforms.Normalize(STANDARD_MEAN, STANDARD_STD))
    elif normalization == "pixelwise":
        transform_list.append(SubtractPixelwiseMean())
    elif normalization == "none":
        pass
    else:
        raise ValueError("Normalization must be 'none', 'standard', or 'pixelwise'")
    
    return transforms.Compose(transform_list)


class UnlabeledDataset(Dataset):
    """
    Wraps a dataset to return only images (dropping labels).
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return image

    def __len__(self):
        return len(self.dataset)
    

def get_train_test_loader(normalization="standard", batch_size=128, num_workers=2, root="./data"):
    transform = get_transform(normalization)
    
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


def get_unsupervised_loader(normalization="standard", batch_size=128, num_workers=2, root="./data"):
    transform = get_transform(normalization)
    
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    unsupervised_dataset = UnlabeledDataset(train_dataset)
    loader = DataLoader(unsupervised_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return loader


class PairDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.augment = transforms.Compose([
            transforms.Pad(2, fill=0),  # Pad 2 pixels on each side to return to 28×28.
            transforms.RandomRotation(25, fill=0),
            transforms.RandomResizedCrop(
                size=28, 
                scale=(0.7, 1.0),    # Crop between 80% and 100% of the original area.
                ratio=(0.75, 1.33)   # Allow aspect ratio changes.
            ),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        # Convert the image to PIL format for augmentation.
        image = transforms.ToPILImage()(image)
        # Create two augmented views of the same image.
        view1 = self.augment(image)
        view2 = self.augment(image)
        return view1, view2

    def __len__(self):
        return len(self.dataset)


def get_ssl_loader(batch_size=1024, num_workers=5, root="./data"):
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    ssl_dataset = PairDataset(train_dataset)
    def collate_fn(batch):
        view1, view2 = zip(*batch)
        view1 = torch.stack(view1) 
        view2 = torch.stack(view2)
        return view1, view2
    ssl_loader = DataLoader(ssl_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    return ssl_loader



if __name__ == '__main__':
    ssl_loader = get_ssl_loader()
    x1, x2 = next(iter(ssl_loader))
    # visualize 4 image pairs below each other i.e. 2 rows and 4 columns
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(4):
        axs[0, i].imshow(x1[i].permute(1, 2, 0).numpy(), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(x2[i].permute(1, 2, 0).numpy(), cmap='gray')
        axs[1, i].axis('off')
    plt.tight_layout()
    plt.show()