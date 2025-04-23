import torch
import torch.nn.functional as F

# this file defines the loss functions used in the experiment

def invariance_loss(z1, z2):
    return F.mse_loss(z1, z2)

def var_loss(z, eps=1e-6):
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(1 - std).mean()

def cov_loss(z):
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (z.size(0) - 1)
    diag = torch.eye(cov.size(0), device=z.device)
    cov_offdiag = cov[~diag.bool()]
    return cov_offdiag.pow(2).sum() / z.size(1)
    

# the standard vicreg loss
# applied to the representations of two subsequent frames, the first term is zero iff the two representations are equal
# i.e. it encourages to extract features that stay constant in time

def vicreg_loss(
    z1, z2,
    inv_weight=25.0,
    var_weight=25.0,
    cov_weight=1.0,
    eps=1e-6,
):
    l_inv = torch.mean((z1 - z2) ** 2)
    l_var = var_loss(z1, eps) + var_loss(z2, eps)
    l_cov = cov_loss(z1) + cov_loss(z2)
    loss = inv_weight * l_inv + var_weight * l_var + cov_weight * l_cov
    return loss, l_inv, l_var, l_cov


# the vicreg loss where l_inv is replaced by a term penalizing the norm of the second derivative
# applied to the representations of three subsequent frames, the first term is zero iff the features between the frames change with a constant rate of change
# i.e. it encourages to extract features that change with a constant derivative

def second_derivative_loss(
    z1, z2, z3,
    inv_weight=25.0, 
    var_weight=25.0, 
    cov_weight=1.0,
    eps=1e-6,
):
    l_inv = torch.mean((z1 - 2 * z2 + z3) ** 2)    
    l_var = var_loss(z1, eps) + var_loss(z2, eps) + var_loss(z3, eps)
    l_cov = cov_loss(z1) + cov_loss(z2) + cov_loss(z3)
    loss = inv_weight * l_inv + var_weight * l_var + cov_weight * l_cov
    return loss, l_inv, l_var, l_cov

