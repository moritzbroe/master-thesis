from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.utils.data import Dataset
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


# this file defines a dataset class that generates images of white circles, triangles and squares on a black background
# in main.py it is only used with a single object per image, but this class can be used to generate images with multiple objects to allow for additional experimentation (e.g. extracting number of objects per class etc...)
# it is easiest to play around with the settings and visualize the results, just run the file and change the dataset parameters at the bottom

CIRCLE = 0
TRIANGLE = 1
SQUARE = 2


# dataset class that will be used for generating unlabeled sequences of images and labeled images

class DynamicObjectDataset(Dataset):
    def __init__(self, num_objects, min_object_size, max_object_size, change_object_size, res, t_vals, boundary_size, fixed_3_objects, unsupervised, label_mode=None):
        '''
        t_vals should be a list of values between 0 and 1. for the unsupervised getitem, images at these values will be generated
        boundary  size is the size of the area outside the unit square (where the image is extracted) where objects can be placed. if set to 0, objects will be placed in the unit square only.
        fixed_3_objects: if True, the dataset will always generate one object of each type. if False, the dataset will generate num_objects objects of random type.
        '''
        self.num_objects = num_objects
        self.min_size = min_object_size
        self.max_size = max_object_size
        self.change_size = change_object_size
        self.res = res
        self.t_vals = t_vals
        self.boundary_size = boundary_size
        self.fixed_3_objects = fixed_3_objects
        self.unsupervised = unsupervised
        self.label_mode = label_mode
        if self.label_mode == 'type':
            assert self.num_objects == 1, "Number of objects must be 1 for type extraction"
        elif self.label_mode == 'position_size':
            assert self.num_objects == 3, "Number of objects must be 3 for position and size extraction"
        elif self.label_mode == 'all':
            assert self.num_objects == 1, "Number of objects must be 1 for all extraction"

    def __len__(self):
        return 10000  # Arbitrary length, irrelevant as data is generated on the fly

    def __getitem__(self, idx):
        if self.unsupervised:
            return self.get_item_unsupervised()
        else:
            return self.get_item_supervised()
    
    def get_item_unsupervised(self, return_position=False):
        # generates sequence of images where objects move with constant speed and size rate of change
        # return_position is used for visualization only

        # Randomly generate initial and final positions, sizes, and types for each object
        positions_0 = [torch.FloatTensor(2).uniform_(-self.boundary_size, 1 + self.boundary_size) for _ in range(self.num_objects)]
        positions_1 = [torch.FloatTensor(2).uniform_(-self.boundary_size, 1 + self.boundary_size) for _ in range(self.num_objects)]

        if self.change_size:
            sizes_0 = [random.uniform(self.min_size, self.max_size) for _ in range(self.num_objects)]
            sizes_1 = [random.uniform(self.min_size, self.max_size) for _ in range(self.num_objects)]
        else:
            sizes_0 = sizes_1 = [random.uniform(self.min_size, self.max_size) for _ in range(self.num_objects)]

        types = [random.choice([CIRCLE, TRIANGLE, SQUARE]) for _ in range(self.num_objects)] if not self.fixed_3_objects else [CIRCLE, TRIANGLE, SQUARE]

        images = []
        for t in self.t_vals:
            image = np.zeros((self.res, self.res), dtype=np.float32)

            # Interpolate position and size for each object
            for i in range(self.num_objects):
                pos = (1 - t) * positions_0[i] + t * positions_1[i]
                size = (1 - t) * sizes_0[i] + t * sizes_1[i]

                # Convert normalized positions to pixel coordinates
                y_pixel = int(pos[0] * self.res)
                x_pixel = int(pos[1] * self.res)
                radius = int(size * self.res / 2)

                # Draw object if any part of it is within the bounds of the unit square
                if (0 <= x_pixel + radius < self.res or 0 <= x_pixel - radius < self.res) and (0 <= y_pixel + radius < self.res or 0 <= y_pixel - radius < self.res):
                    if types[i] == CIRCLE:
                        self._draw_circle(image, x_pixel, y_pixel, radius)
                    elif types[i] == TRIANGLE:
                        self._draw_triangle(image, x_pixel, y_pixel, radius)
                    elif types[i] == SQUARE:
                        self._draw_square(image, x_pixel, y_pixel, radius)

            images.append(torch.tensor(image, dtype=torch.float).unsqueeze(0))  # 1 x res x res

        images = torch.stack(images)

        if return_position: # only used for visualization
            assert self.num_objects == 1, "return_positions only works for 1 object"
            return images, [(1-t) * positions_0[0] + t * positions_1[0] for t in self.t_vals], [(1-t) * sizes_0[0] + t * sizes_1[0] for t in self.t_vals]
        else:
            return images
    
    def get_item_supervised(self):
        # generate just one image and return image and object positions
        positions = [torch.FloatTensor(2).uniform_(-self.boundary_size, 1 + self.boundary_size) for _ in range(self.num_objects)]
        sizes = [random.uniform(self.min_size, self.max_size) for _ in range(self.num_objects)]
        types = [random.choice([CIRCLE, TRIANGLE, SQUARE]) for _ in range(self.num_objects)] if not self.fixed_3_objects else [CIRCLE, TRIANGLE, SQUARE]

        image = np.zeros((self.res, self.res), dtype=np.float32)

        # Interpolate position and size for each object
        for i in range(self.num_objects):
            pos = positions[i]
            size = sizes[i]

            # Convert normalized positions to pixel coordinates
            y_pixel = int(pos[0] * self.res)
            x_pixel = int(pos[1] * self.res)
            radius = int(size * self.res / 2)

            # Draw object if any part of it is within the bounds of the unit square
            if (0 <= x_pixel + radius < self.res or 0 <= x_pixel - radius < self.res) and (0 <= y_pixel + radius < self.res or 0 <= y_pixel - radius < self.res):
                if types[i] == CIRCLE:
                    self._draw_circle(image, x_pixel, y_pixel, radius)
                elif types[i] == TRIANGLE:
                    self._draw_triangle(image, x_pixel, y_pixel, radius)
                elif types[i] == SQUARE:
                    self._draw_square(image, x_pixel, y_pixel, radius)
                    
        if self.label_mode == 'type':   # return for each image the type of the one object in it
            label = types[0]
        elif self.label_mode == 'count':    # return for each image a 3-tuple with the count of each object type visible in the image. can be used to get one-hot encodings when using only one object which is always visible.
            res = [0, 0, 0]
            for i in range(self.num_objects):
                if 0 <= positions[i][0] <= 1 and 0 <= positions[i][1] <= 1:
                    res[types[i]] += 1
            label = torch.tensor(res, dtype=torch.float)
        elif self.label_mode == 'position_size': # return position and size of the 3 objects in each image, set size to zero if object is not in image
            for i in range(3):
                if positions[i][0] < 0 or positions[i][0] > 1 or positions[i][1] < 0 or positions[i][1] > 1:
                    positions[i] = torch.zeros(2)
                    sizes[i] = 0
            label = torch.tensor([item for i in range(3) for item in [positions[i][0].item(), positions[i][1].item(), sizes[i]]])
        elif self.label_mode == 'all': 
            if positions[0][0] < 0 or positions[0][0] > 1 or positions[0][1] < 0 or positions[0][1] > 1:
                positions[0] = torch.zeros(2)
                sizes[0] = 0
            label = torch.tensor([types[0], positions[0][0].item(), positions[0][1].item(), sizes[0]])
        else:
            raise ValueError(f"Invalid extract value {self.label_mode}. Must be 'type', 'count', or 'position_size'.")

        return torch.tensor(image).unsqueeze(0), label

    def _draw_circle(self, image, x, y, radius):
        y_grid, x_grid = np.ogrid[:self.res, :self.res]
        mask = (x_grid - x) ** 2 + (y_grid - y) ** 2 <= radius ** 2
        image[mask] = 1.0

    def _draw_triangle(self, image, x, y, radius):
        vertices = [
            (x, y - radius),
            (x - radius, y + radius),
            (x + radius, y + radius)
        ]
        self._draw_polygon(image, vertices)

    def _draw_square(self, image, x, y, radius):
        vertices = [
            (x - radius, y - radius),
            (x - radius, y + radius),
            (x + radius, y + radius),
            (x + radius, y - radius)
        ]
        self._draw_polygon(image, vertices)

    def _draw_polygon(self, image, vertices):
        from skimage.draw import polygon
        vertices = np.array(vertices)
        rr, cc = polygon(vertices[:, 1], vertices[:, 0], image.shape)
        image[rr, cc] = 1.0


# this function creates the data loaders for the dataset. only need ssl and train dataset, as train, val and test are all the same (as data is generated on the fly)
def get_data_loaders_shapes(batch_size=128, num_workers=10, resolution=64, num_objects=1, min_object_size=0.1, max_object_size=0.3, change_size=True, t_vals=(0,1), boundary_size=0, fixed_3_objects=False, label_mode='all'):
    if not fixed_3_objects:
        assert num_objects is not None, "Number of objects must be specified when not using fixed_3_objects"
    if fixed_3_objects:
        num_objects = 3
    # create ssl loader
    ssl_dataset = DynamicObjectDataset(num_objects=num_objects, min_object_size=min_object_size, max_object_size=max_object_size, change_object_size=change_size, res=resolution, t_vals=t_vals, boundary_size=boundary_size, fixed_3_objects=fixed_3_objects, unsupervised=True)
    train_dataset = DynamicObjectDataset(num_objects=num_objects, min_object_size=min_object_size, max_object_size=max_object_size, change_object_size=change_size, res=resolution, t_vals=t_vals, boundary_size=boundary_size, fixed_3_objects=fixed_3_objects, unsupervised=False, label_mode=label_mode)
    ssl_loader = DataLoader(ssl_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return ssl_loader, train_loader


# visualize a sample from a dataset
def visualize_sample(dataset):
    images = dataset[0]
    if isinstance(images, list):
        images, labels = images
        plt.imshow(images.numpy().transpose(1, 2, 0), cmap='gray')
        plt.title(labels)
        plt.show()
    else:
        num_images = images.shape[0]
        fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
        if num_images == 1:
            axes = [axes]
        for ax, img in zip(axes, images):
            ax.imshow(img.numpy().transpose(1, 2, 0), cmap='gray')
    
        plt.tight_layout()
        plt.show()


# display a triple with positions and sizes, works only for a dataset with 1 object
def visualize_labeled_triple(dataset):
    images, positions, sizes = dataset.get_item_unsupervised(return_position=True)
    
    num_images = images.shape[0]
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
    if num_images == 1:
        axes = [axes]

    # pad images with a single grey pixel border
    images = F.pad(images, (1, 1, 1, 1), mode='constant', value=0.5)
    
    # Define a font dictionary for a monospace font.
    monospace_font = {'family': 'monospace', 'fontsize': 14}
    
    for i, ax in enumerate(axes):
        # Remove the channel dimension and display the image.
        img = images[i].squeeze(0).numpy()
        ax.imshow(img, cmap='gray')
        
        # Format text with .3f.
        pos_line = f"Position = ({positions[i][0]:.3f}, {positions[i][1]:.3f})"
        size_line = f"Size     = {sizes[i]:.3f}         "
        label = pos_line + "\n" + size_line
        
        # Set the title with a monospace font so the equal signs align.
        ax.set_title(label, fontdict=monospace_font)
        ax.axis("off")

    plt.show()




if __name__ == '__main__':
    # Test your dataset and visualize one triple with positions and sizes.
    dataset = DynamicObjectDataset(num_objects=5, min_object_size=0.1, max_object_size=0.3, change_object_size=True, res=256, t_vals=(0, 0.2, 0.4, 0.6, 0.8, 1), boundary_size=0.5, fixed_3_objects=False, unsupervised=True)
    visualize_sample(dataset)

