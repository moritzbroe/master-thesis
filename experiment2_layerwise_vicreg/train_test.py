import torch
import torch.nn as nn
import torch.optim as optim
from loss import vicreg_loss
from linear_probe import linear_probe, linear_probe_cross_validated
from ssl_loader import get_basic_ssl_loader, set_transformations
import numpy as np


# This is the most important part of the code, defining the functions for blockwise/layerwise training with different augmentations 


# trains the model's layers from start_layer to final_layer with the vicreg loss and the given self-supervised learning loader that gives augmented pairs of images
def train_model_layer(model, start_layer, final_layer, ssl_loader, inv_coeff, epochs=1, verbose=True, lr=1e-3, device='cuda:0', clip_grads=None, warmup_steps=0):
    if verbose:
        print('train layer', start_layer)
    # only add the parameters of the block to the optimizer
    params = []
    for layer in range(start_layer, final_layer + 1):
        params = params + list(model[layer].parameters())
    optimizer = optim.AdamW(params, lr=lr)

    # simple lr warmup
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps))

    scaler = torch.amp.GradScaler(device)
    for epoch in range(epochs):
        for step, (x1, x2) in enumerate(ssl_loader):
            x1 = x1.to(device)
            x2 = x2.to(device)

            # first pass inputs through the part of the model before start_layer without gradient tracking
            with torch.autocast(device_type=device, dtype=torch.float16):
                with torch.no_grad():
                    inputs1 = model[:start_layer](x1)
                    inputs2 = model[:start_layer](x2)
                    inputs1 = inputs1.detach()
                    inputs2 = inputs2.detach()

            # now through the relevant layers with gradient tracking. no need to pass through the layers after that.
            with torch.autocast(device_type=device, dtype=torch.float16):
                outputs1 = model[start_layer : final_layer + 1](inputs1)
                outputs2 = model[start_layer : final_layer + 1](inputs2)

                loss, l_inv, l_var, l_cov = vicreg_loss(
                    outputs1, outputs2, var_weight=25.0, cov_weight=1.0, inv_weight=inv_coeff,
                    mode='full'
                )

            # adapt parameters
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip_grads: # gradient clipping for stability 
                nn.utils.clip_grad_norm_(params, clip_grads)
            scaler.step(optimizer)
            scaler.update()
            if warmup_steps > 0:
                scheduler.step()

            if step % 20 == 0 and verbose:
                print(f"  [Layer={final_layer}] Epoch {epoch}, Step {step}/{len(ssl_loader)}: "
                      f"Loss={loss.item():.4f}, Inv={l_inv.item():.4f}, "
                      f"Var={l_var.item():.4f}, Cov={l_cov.item():.4f}")



# Full precision version of the function above

# def train_model_layer(model, start_layer, final_layer, ssl_loader, epochs=1, verbose=True, lr=1e-3, device='cuda', clip_grads=None, inv_coeff=None, warmup=0):
#     print('train layer', start_layer)
#     params = []
#     for layer in range(start_layer, final_layer + 1):
#         params = params + list(model[layer].parameters())
#     optimizer = optim.AdamW(params, lr=lr)

#     # simple lr warmup
#     if warmup > 0:
#         scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup))

#     for epoch in range(epochs):
#         for step, (x1, x2) in enumerate(ssl_loader):
#             x1 = x1.to(device)
#             x2 = x2.to(device)

#             with torch.no_grad():
#                 inputs1 = model[:start_layer](x1)
#                 inputs2 = model[:start_layer](x2)
#                 inputs1 = inputs1.detach()
#                 inputs2 = inputs2.detach()

#             outputs1 = model[start_layer : final_layer + 1](inputs1)
#             outputs2 = model[start_layer : final_layer + 1](inputs2)

#             loss, l_inv, l_var, l_cov = vicreg_loss(
#                 outputs1, outputs2, var_weight=25.0, cov_weight=1.0, inv_weight=inv_coeff,
#                 mode='full'
#             )

#             optimizer.zero_grad()
#             loss.backward()
#             if clip_grads:
#                 nn.utils.clip_grad_norm_(params, clip_grads)
#             optimizer.step()
#             if warmup > 0:
#                 scheduler.step()

#             if step % 20 == 0 and verbose:
#                 print(f"  [Layer={final_layer}] Epoch {epoch}, Step {step}/{len(ssl_loader)}: "
#                       f"Loss={loss.item():.4f}, Inv={l_inv.item():.4f}, "
#                       f"Var={l_var.item():.4f}, Cov={l_cov.item():.4f}")
                



def test_params(model, loss_layers, params, C=0.01, verbose=True, epochs=1,
                device='cuda:0', train_loader=None, test_loader=None, ssl_loader=None, k_cross_val=None, clip_grads=None, resolution=96, warmup_steps=0):
    '''
    This function does the following. it takes a model which is a nn.Sequential and a list loss_layers of layers to apply the loss after. 
    moreover, params is a list where the i'th entry is a list of parameters for the i'th layer. 
    this list contains the augmentation parameters min_size (for cropping), color_jitter_strength, flip probability, and the learning_rate and invariance coefficient used for the vicreg loss.
    these 5 values are provided for each layer as each layer can have different augmentations etc.
    The function trains the model for the given number of epochs and returns the accuracy where the linear probe accuracy, which is either computed 
    via cross-validation on the train set (for hyperparamter tuning) or by training on the train set and testing on the test set (for final evaluation).
    '''
    if verbose:
        print('TEST PARAMS')
    assert len(loss_layers) == len(params), "loss_layers and params must have the same length"
    if k_cross_val is None:
        assert test_loader is not None, "test_loader must be provided if k_cross_val is None"
    if k_cross_val is not None and test_loader is not None:
        print("Warning: test_loader will be ignored since k_cross_val is provided.")
    # now go through the layers and train each one with the correct parameters for the given number of epochs
    for i in range(len(loss_layers)):
        start_layer = loss_layers[i-1] + 1 if i > 0 else 0
        final_layer = loss_layers[i]
        # set the transformations to the correct ones for the layer
        ssl_loader = set_transformations(ssl_loader, color_jitter_strength=params[i][1], min_size=params[i][0], flip=params[i][2], res=resolution)
        train_model_layer(
            model,
            start_layer, final_layer, 
            ssl_loader,
            lr=params[i][3],
            epochs=epochs,
            verbose=verbose,
            device=device,
            clip_grads=clip_grads,
            inv_coeff=params[i][4],
            warmup_steps=warmup_steps,
        )
    if k_cross_val is not None:
        test_acc = linear_probe_cross_validated(model, train_loader, n_folds=k_cross_val, device=device, verbose=verbose, C=C)
    else:
        test_acc = linear_probe(model, train_loader, test_loader, device=device, verbose=verbose, C=C)

    return test_acc
