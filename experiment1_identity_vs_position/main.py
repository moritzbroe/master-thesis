from shapes import get_data_loaders_shapes, visualize_sample
import torch.nn as nn
from train_test import train_vicreg, train_triple, test_class, test_position
import numpy as np 

# This script runs the VicReg and Triple loss experiments on the shapes dataset and shows 
# that training with vicreg makes the feature extractor encode object identity in its output, 
# while training with the second derivative penalization makes it encode positional information.
# takes some time to run: had to train for 30 epochs in order for the models to actually always 
# converge to the "correct" solution, especially for the classification...

# create dataloaders for pair and triples used for vicreg and triple loss respectively and the supervised loader that also returns position/size/class information
# note that the default arguments make these dataloaders generate images sequences of a single object

ssl_loader_vicreg, supervised_loader = get_data_loaders_shapes(batch_size=512, t_vals=(0, 1))
ssl_loader_triple, _ = get_data_loaders_shapes(batch_size=512, t_vals=(0, 0.5, 1.0))

# # uncomment to visualize a single sample sequence from each dataloader
# visualize_sample(ssl_loader_vicreg.dataset)
# visualize_sample(ssl_loader_triple.dataset)


# simple cnn model
model_fn = lambda: nn.Sequential(
    nn.Conv2d(1, 16, 5, 1, 2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 5, 1, 2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 5, 1, 2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(8*8*64, 3),
).to('cuda')


print('UNTRAINED MODEL')
# test performance of the untrained model for linear extractability of position and class information
initial_class_acc = [test_class(model_fn(), supervised_loader) for _ in range(10)]
initial_pos_l2 = [test_position(model_fn(), supervised_loader) for _ in range(10)]

print()

# now for several iterations train the model with vicreg and check linear extractability of position and class information
print('VICREG TRAINED MODEL')
vicreg_class_acc = []
vicreg_pos_l2 = []
for _ in range(10):
    model = model_fn()
    train_vicreg(model, ssl_loader_vicreg, num_epochs=30, verbose=False)
    vicreg_class_acc.append(test_class(model, supervised_loader))
    vicreg_pos_l2.append(test_position(model, supervised_loader))

print()

# the same for the triple loss
print('TRIPLE LOSS TRAINED MODEL')
triple_class_acc = []
triple_pos_l2 = []
for _ in range(10):
    model = model_fn()
    train_triple(model, ssl_loader_triple, num_epochs=30, verbose=False)
    triple_class_acc.append(test_class(model, supervised_loader))
    triple_pos_l2.append(test_position(model, supervised_loader))


print('SUMMARY')
print('INITIAL CLASS ACCURACY: MEAN:', np.mean(initial_class_acc), 'STD:', np.std(initial_class_acc))
print('INITIAL POSITION L2 ERROR: MEAN:', np.mean(initial_pos_l2), 'STD:', np.std(initial_pos_l2))
print('VICREG CLASS ACCURACY: MEAN:', np.mean(vicreg_class_acc), 'STD:', np.std(vicreg_class_acc))
print('VICREG POSITION L2 ERROR: MEAN:', np.mean(vicreg_pos_l2), 'STD:', np.std(vicreg_pos_l2))
print('TRIPLE CLASS ACCURACY: MEAN:', np.mean(triple_class_acc), 'STD:', np.std(triple_class_acc))
print('TRIPLE POSITION L2 ERROR: MEAN:', np.mean(triple_pos_l2), 'STD:', np.std(triple_pos_l2))
