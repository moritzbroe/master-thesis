from torch import nn
from models.models_fully_local import MLPLayer, LocalSequential, Conv2DLayer
from data.mnist_datasets import get_train_test_loader, get_unsupervised_loader, get_ssl_loader
import torch.nn.functional as F
from initializers.initializer import Initializer, MultiLayerInitializer
from learning_rules.fully_local_learning_rules import generate_exponents_total_degree, PolynomialLearningRule, MultiLayerLearningRule, PolynomialLearningRulePair
from train_test.hypgraddesc import hypgraddesc
import matplotlib.pyplot as plt
from train_test.linear_probe import linear_probe
import argparse
import numpy as np


# this is the main method. it applies fully local learning rules to an mlp and updates the initializer's and learning rule's hyperparameters to minimize the linear classification loss.
# in the thesis the following commands were used:
# python3 main.py 32 --lr 1e-5 --metasteps 2000                   -> single layer with 32 output neurons trained on unsupervised data
# python3 main.py 256 32 --lr 1e-5 --metasteps 2000               -> two layer mlp with 256, 32 neurons trained on unsupervised data
# python3 main.py 32 --pair_inputs --lr 1e-5 --metasteps 1000     -> single layer with 32 output neurons trained on ssl data (augmented image pairs)
# python3 main.py 256 32 --pair_inputs --lr 1e-5 --metasteps 1000 -> two layer mlp with 256, 32 neurons trained on ssl data (augmented image pairs)


parser = argparse.ArgumentParser()

parser.add_argument("--pixelwise_normalization", action="store_true", help="Enable pixelwise normalization")
parser.add_argument("layers", type=int, nargs="+", help="Layer sizes")
parser.add_argument("--l1_reg", type=float, default=0.0, help="L1 regularization for learning rule")
parser.add_argument("--sparsify", type=int, default=None, help="Sparsify from metastep")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for the learning rule")
parser.add_argument("--max_inner_steps", type=int, default=250, help="Max inner steps for the learning rule")
parser.add_argument("--raise_every", type=int, default=100, help="Raise every n steps")
parser.add_argument("--raise_by", type=int, default=0, help="Raise by n steps")
parser.add_argument("--initial_inner_steps", type=int, default=250, help="Initial steps for the learning rule")
parser.add_argument("--C", type=float, default=1000.0, help="C for linear probe")
parser.add_argument("--bound", type=float, default=0.1)
parser.add_argument("--pair_inputs", action="store_true", help="Pair inputs for the learning rule")
parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps for the learning rule")
parser.add_argument("--metasteps", type=int, default=1000, help="Meta steps for the learning rule")
parser.add_argument("--finetune_steps", type=int, default=0, help="Finetune steps for the learning rule")
parser.add_argument("--clip_grads", type=float, default=None, help="Clip gradients")
parser.add_argument("--inh_steps", type=int, default=10, help="Inhibition steps")
parser.add_argument("--inh_factor", type=float, default=0.2, help="Inhibition factor")
parser.add_argument("--l2_reg", type=float, default=0.0, help="L2 regularization for updates")
parser.add_argument("--optimizer", type=str, default='adam', help="Optimizer for the learning rule")

args = parser.parse_args()

layer_sizes = [784] + args.layers
pixelwise_normalization = args.pixelwise_normalization

device='cuda:0'


# define learning rule and dataloader depending on whether or not to use pair_inputs, i.e. augmented image pairs, for the inner update steps
if args.pair_inputs:
    if pixelwise_normalization:
        raise NotImplementedError("Pixelwise normalization not implemented for pair inputs")
    unsup_loader = get_ssl_loader()
    learning_rule = MultiLayerLearningRule(
    learning_rules=[
        PolynomialLearningRulePair(
            lat_inh=True,
            exponents_m=[e for e in generate_exponents_total_degree(5, 3) if e <= (e[2], e[3], e[0], e[1], e[4])],  # allow polynomials of degree at most 3, but remove reduncancy in the exponents to reduce number of hyperparameters
            exponents_w=[e for e in generate_exponents_total_degree(7, 3) if e <= (e[3], e[4], e[5], e[0], e[1], e[2], e[6])],
            exponents_b=[e for e in generate_exponents_total_degree(3, 3) if e <= (e[1], e[0], e[2])],
            bound='tanh ' + str(args.bound),
            m_diagonal_component=True,
            l2_reg=args.l2_reg,
            forward_weight_bound=1.0,
        ) for _ in range(len(layer_sizes) - 1)
    ]
).to(device)

else:
    unsup_loader = get_unsupervised_loader(normalization='pixelwise' if pixelwise_normalization else 'none', batch_size=1024)
    learning_rule = MultiLayerLearningRule(
        learning_rules=[
            PolynomialLearningRule(
                lat_inh=True,
                exponents_m=[e for e in generate_exponents_total_degree(3, 3)],
                exponents_w=[e for e in generate_exponents_total_degree(4, 3)],
                exponents_b=[e for e in generate_exponents_total_degree(2, 3)],
                bound='tanh ' + str(args.bound),
                m_diagonal_component=True,
                forward_weight_bound=1.0,
                l2_reg=args.l2_reg,
            ) for _ in range(len(layer_sizes) - 1)
        ]
    ).to(device)


train_loader, test_loader = get_train_test_loader(normalization='pixelwise' if pixelwise_normalization else 'none', batch_size=1024)
# initializer initializes forward weights with mean 0 and learnable std, bias and lateral weights as learnable constants
initializer = MultiLayerInitializer(initializer_list=[Initializer(weights='std', bias='mean', lateral='mean', weight_init_trainable=True, bias_init_trainable=True, lateral_init_trainable=True) for _ in range(len(layer_sizes) - 1)]).to(device)

# model uses lateral inhibition, consists of flattening and mlp layers
def model_fn():
    layers = [nn.Flatten(start_dim=1, end_dim=-1)]
    for i in range(len(layer_sizes) - 1):
        layers.append(MLPLayer(
            layer_sizes[i], layer_sizes[i+1], 
            activation=lambda x: F.sigmoid(4*x),
            inh_mode='weights',
            no_anti_inhibition=False,
            no_self_inhibition=False,
            inhibition_steps=args.inh_steps,
            inhibition_factor=args.inh_factor,
        ))
    return LocalSequential(layers)

# collect test accs and losses for plotting
test_accs = []
losses = []

# loss function uses linear evaluation performance
def loss_fct(model, params):
    loss, acc_train, acc_test = linear_probe(model, params, train_loader, test_loader, C=args.C, device=device, loss_batch_size=1024, use_test_loader=False,)
    test_accs.append(acc_test)
    losses.append(loss.item())
    print(f"Loss: {loss.item()}")
    # print mean and std of parameters to see behavior
    print("FINAL MEAN+STD" + "==" * 30)
    for i, layer in enumerate(params):
        print('LAYER', i)
        for k, v in layer.items():
            print(k, 'mean:', v.mean().item(), 'std:' ,v.std().item(), 'max:', v.max().item(),'min:', v.min().item())
    # print some outputs
    print("SAMPLE OUTPUTS" + "==" * 30)
    x = next(iter(unsup_loader))
    if args.pair_inputs:
        x0 = x[0].to(device)
        x1 = x[1].to(device)
        y0 = model(x0, params)
        y1 = model(x1, params)
        print(y0[:2])
        print(y1[:2])
    else:
        x = x.to(device)
        out = model(x, params)
        print(out[:2])
    print("==" * 30)
    if acc_test < 0.5:
        print("Test accuracy too low, stopping training")
        raise ValueError("Test accuracy too low")
    return loss

# run the hypergradient descent procedure
hypgraddesc(
    model_fn=model_fn,
    learning_rule=learning_rule,
    initializer=initializer,
    unsup_loader=unsup_loader,
    loss_fct=loss_fct,
    train_initializer=True,
    inner_steps=lambda ms: min(args.max_inner_steps, args.initial_inner_steps + args.raise_by * (ms // args.raise_every)),
    meta_steps=args.metasteps,
    meta_lr=args.lr,
    lr_warmup=args.warmup_steps,
    clip_grads=args.clip_grads,
    sparsify=args.sparsify,
    l1_reg_learning_rule=args.l1_reg,
    device=device,
    accumulate_updates=True,
    finetune_steps=args.finetune_steps,
    finetune_lr_factor=0.2,
    optimizer_name=args.optimizer,
)

# plot losses and test accuracies, should even run when stopping with ctrl+c
print(losses)
print(test_accs)
plt.plot(losses)
plt.title('Loss')
plt.show()
plt.plot(test_accs)
plt.title('Test Accuracy')
plt.show()