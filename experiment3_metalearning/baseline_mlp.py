
from torch import nn
from models.models_fully_local import MLPLayer, LocalSequential, Conv2DLayer
from data.mnist_datasets import get_train_test_loader, get_unsupervised_loader
import torch
import torch.nn.functional as F
from initializers.initializer import Initializer
from learning_rules.fully_local_learning_rules import PolynomialLearningRule
from train_test.hypgraddesc import hypgraddesc
from train_test.train import train
import matplotlib.pyplot as plt
from train_test.linear_probe import linear_probe
from math import sqrt
import sys

# this sets a baseline by considering feature extractors consisting of a randomly initialized fully connected layer with sigmoid activation
# weights and biases are initialized with mean and std that are trainable parameters and only these parameters are trained to minimize logistic regression loss.

# number of output neurons, i.e. representation size
k = 10000
# regularization parameter C for logistic regression, could be further tuned to improve performance, currently using:
C = 1 if k < 1000 else 0.3

device='cuda'

# create initializer which intializes weights and biases with mean and std that are trainable parameters
initializer = Initializer(bias_scaler=1, weight_scaler=1, weights='normal', bias='normal', weight_init_trainable=True, bias_init_trainable=True)

unsup_loader = get_unsupervised_loader() # not used, but needs to be iterable...
train_loader, test_loader = get_train_test_loader(normalization='pixelwise', batch_size=1024)

learning_rule = PolynomialLearningRule() # not used either

model_fn = lambda: LocalSequential([
    nn.Flatten(start_dim=1, end_dim=-1), 
    MLPLayer(
        784, k, 
        activation=nn.Sigmoid(),
        inh_mode=None,
    )
])

test_accs = []
# define differentiable loss function, which is just the linear probe loss used almost throughout.
def loss_fct(model, params):
    loss, acc_train, acc_test = linear_probe(model, params, train_loader, test_loader, grad=True, device=device, loss_batch_size=1024, C=C, use_test_loader=False)
    test_accs.append(acc_test)
    print(f"Loss: {loss.item()}")
    return loss


# now run the hypergradient descent procedure to train the initializer
hypgraddesc(
    model_fn=model_fn,
    learning_rule=learning_rule,
    initializer=initializer,
    unsup_loader=unsup_loader,
    loss_fct=loss_fct,
    train_initializer=True,
    inner_steps=lambda ms: 0, # train only initializer: 0 inner update steps are performed, so the learning rule is not used
    initial_steps=0,
    meta_steps=200,
    meta_lr=0.1,
    lr_warmup=10,
    device=device,
    l2_reg_learning_rule=0.0,
    finetune_steps=100, # reduce learning rate for the last 100 steps by factor 10
    finetune_lr_factor=0.1,
)


print(test_accs)
plt.plot(test_accs)
plt.title('Test Accuracy')
plt.show()

