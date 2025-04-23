
from itertools import cycle
from torch.utils.checkpoint import checkpoint
import torch


# applies a learning rule to update parameters of a model for several steps
# when setting grad=True, a computational graph will be built with all intermediate params - i.e. all the updates are unrolled.
# importantly, using checkpointing is recommended as it severely reduces memory usage

def train(model, params, loader, learning_rule, steps=1, device='cuda', grad=True, use_checkpointing=True):
    # move params to device, a bit ugly...
    params = [{k: v.to(device) for k, v in p.items()} for p in params]

    i = 0
    while True:
        for x in loader:
            if i == steps:
                return params
            x = [xi.to(device) for xi in x] if isinstance(x, (list, tuple)) else x.to(device)   # x can be list (for ssl (or even supervised learning)) or tensor (for standard unsupervised learning)
            if grad:
                if use_checkpointing:
                    params = checkpoint(learning_rule.update, model, params, x, use_reentrant=False)
                else:
                    params = learning_rule.update(model, params, x)
            else:
                params = learning_rule.update(model, params, x)
                params = [{k: v.detach().requires_grad_() for k, v in layer_params.items()} for layer_params in params]
            i += 1
            
