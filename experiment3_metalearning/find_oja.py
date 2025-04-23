
from torch import nn
from models.models_fully_local import MLPLayer, LocalSequential
from data.mnist_datasets import get_unsupervised_loader
import torch
from initializers.initializer import Initializer
from learning_rules.fully_local_learning_rules import PolynomialLearningRule, generate_exponents_partial_degree
from train_test.hypgraddesc import hypgraddesc
import matplotlib.pyplot as plt

# this experiment shows how to (almost) recover oja's rule with the hypergradient descent procedure

device = 'cuda'
# use pixelwise centering here
unsup_loader = get_unsupervised_loader(normalization='pixelwise', batch_size=1024)


# manually extract first principal direction from mnist
mnist_data = []
for x in unsup_loader:
    mnist_data.append(x)
mnist_data = torch.cat(mnist_data, dim=0)
mnist_data = mnist_data.view(mnist_data.shape[0], -1)
cov = torch.matmul(mnist_data.T, mnist_data) / mnist_data.shape[0]
eigvals, eigvecs = torch.linalg.eigh(cov)
first_pc = eigvecs[:,-1].to(device)

# use standard initializer, initializing weights with std=1
initializer = Initializer(weights='std', weight_init_trainable=False)

# learning rule uses polynomials where each exponent is at most 2, and removes all exponent tuples where the pre-activation exponent is non-zero, hence not allowing dependence of updates on pre-activations
learning_rule = PolynomialLearningRule(
    exponents_w=[e for e in generate_exponents_partial_degree(4, 2) if e[1] == 0],
    exponents_b=[],
    bound='tanh 0.1',
)

# single linear neuron model
model_fn = lambda: LocalSequential([nn.Flatten(start_dim=1, end_dim=-1), MLPLayer(784, 1, activation=nn.Identity(), inh_mode=None)])

losses = []
# loss funcition is the squared distance to the first principal direction (the closer one of the two version differing by a sign)
def loss_fct(model, params):
    dist1 = torch.norm(params[1]['w'] - first_pc)**2
    dist2 = torch.norm(params[1]['w'] + first_pc)**2
    loss = torch.min(dist1, dist2)
    print(f"Loss: {loss.item()}")
    print(f"Norm: {torch.norm(params[1]['w'])}")
    losses.append(loss.item())
    return loss


# run hypergradient descent
hyperparams = hypgraddesc(
    model_fn=model_fn,
    learning_rule=learning_rule,
    initializer=initializer,
    unsup_loader=unsup_loader,
    loss_fct=loss_fct,
    train_initializer=False,
    inner_steps=300,
    initial_steps=0,
    reinitialize_every_steps=1,
    meta_steps=500,
    meta_lr=1e-4,
    lr_warmup=10,
    clip_grads=1000.0,
    l1_reg_learning_rule=100.0, # use l1 regularization to keep most hyperparams at zero
    plot_hyperparams=lambda lr: lr.params_w,   # plot the learning rule parameters
    device=device,
    accumulate_updates=True,
)

plt.plot(losses)
plt.show()

# can plot hyperparams here...