import torch
import torch.nn.functional as F

# this defines the vicreg losses for convolutional feature maps (but also works for 2D tensors)

def invariance_loss(z1, z2):
    return F.mse_loss(z1, z2)

def var_loss(z, eps=1e-6):
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(1 - std).mean()

def cov_loss(z):
    if z.dim() == 2:
        # Original implementation for [B, C] inputs.
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (z.size(0) - 1)
        diag = torch.eye(cov.size(0), device=z.device)
        cov_offdiag = cov[~diag.bool()]
        return cov_offdiag.pow(2).sum() / z.size(1)
    elif z.dim() == 4:
        B, C, H, W = z.shape
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov = torch.einsum('bchw,bdhw->cdhw', z_centered, z_centered) / (B - 1)
        eye = torch.eye(C, device=z.device).view(C, C, 1, 1)
        cov_offdiag = cov * (1 - eye)
        return (cov_offdiag.pow(2).sum(dim=(0, 1)) / C).mean()
    else:
        raise ValueError("Input tensor z must be of shape [B, C] or [B, C, H, W].")


def vicreg_loss(
    z1, z2, 
    inv_weight=25.0, 
    var_weight=25.0, 
    cov_weight=1.0,
    eps=1e-6,
    mode='full',
):
    '''the vicreg loss function. global average pooling mode (gap) averages over spatial dimensions, then applies vicreg loss. 'full' mode applies vicreg loss to each spatial dimension, then averages over spatial dimensions.'''
    assert z1.shape == z2.shape
    if mode == 'gap':
        if z1.dim() == 4:
            z1 = F.adaptive_avg_pool2d(z1, 1).flatten(1)
            z2 = F.adaptive_avg_pool2d(z2, 1).flatten(1)
    elif mode == 'full':
        pass
    else:
        raise ValueError('Invalid mode')
    l_inv = invariance_loss(z1, z2)
    l_var = var_loss(z1, eps) + var_loss(z2, eps)
    l_cov = cov_loss(z1) + cov_loss(z2)
    loss = inv_weight * l_inv + var_weight * l_var + cov_weight * l_cov
    return loss, l_inv, l_var, l_cov