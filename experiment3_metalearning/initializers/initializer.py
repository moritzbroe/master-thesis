import torch
from torch import nn
from math import sqrt

# initializers are trainable pytorch modules, which can be called to get the initial weights and biases for a layer
# various parameters for initialization can be set to be trainable or fixed

class Initializer(nn.Module):
    def __init__(self, bias_scaler=100, lateral_scaler=100, weight_scaler=1000, weights='std', bias=None, lateral=None, weight_init_trainable=False, bias_init_trainable=False, lateral_init_trainable=False):
        # the scalers effectively scale the learning rate for the initialization parameters, as these generally need higher learning rates than the learning rule's params. hacky solution and should be improved...
        super(Initializer, self).__init__()
        self.w_mean = torch.nn.Parameter(torch.tensor(0.0), requires_grad=weight_init_trainable) if weights in ['mean', 'normal'] else 0.0
        self.b_mean = torch.nn.Parameter(torch.tensor(0.0), requires_grad=bias_init_trainable) if bias in ['mean', 'normal'] else 0.0
        self.m_mean = torch.nn.Parameter(torch.tensor(0.0), requires_grad=lateral_init_trainable) if lateral in ['mean', 'normal'] else 0.0
        self.w_std = torch.nn.Parameter(torch.tensor(1.0 / weight_scaler), requires_grad=weight_init_trainable) if weights in ['std', 'normal'] else 0.0
        self.b_std = torch.nn.Parameter(torch.tensor(1.0 / bias_scaler), requires_grad=bias_init_trainable) if bias in ['std', 'normal'] else 0.0
        self.m_std = torch.nn.Parameter(torch.tensor(1.0 / lateral_scaler), requires_grad=lateral_init_trainable) if lateral in ['std', 'normal'] else 0.0
        self.bias_scaler = bias_scaler
        self.lateral_scaler = lateral_scaler
        self.weight_scaler = weight_scaler

    def get_forward_weights(self, out_dim, in_dim):
        # weight initialization std is also scaled by 1/sqrt(input_dim) to account for the number of inputs to the neuron
        # should possibly do something similar for the lateral weights?
        device = self._get_device()
        return self.weight_scaler / sqrt(in_dim) * (self.w_mean * torch.ones((out_dim, in_dim), requires_grad=True, device=device) + self.w_std * torch.randn((out_dim, in_dim), requires_grad=True, device=device))
    
    def get_biases(self, out_dim):
        device = self._get_device()
        return self.bias_scaler * (self.b_mean * torch.ones((out_dim,), requires_grad=True, device=device) + self.b_std * torch.randn((out_dim,), requires_grad=True, device=device))
    
    def get_lateral_weights(self, out_dim):
        device = self._get_device()
        return self.lateral_scaler * (self.m_mean * torch.ones((out_dim, out_dim), requires_grad=True, device=device) + self.m_std * torch.randn((out_dim, out_dim), requires_grad=True, device=device))
    
    def get_forward_conv_weights(self, out_channels, in_channels, kernel_size):
        raise NotImplementedError

    def __str__(self):
        item = lambda x: x.item() if isinstance(x, torch.Tensor) else x
        return (
            f"\nw_mean={item(self.w_mean) * self.weight_scaler}, w_std={item(self.w_std) * self.weight_scaler}/sqrt(input_dim),\n "
            f"b_mean={item(self.b_mean) * self.bias_scaler}, b_std={item(self.b_std) * self.bias_scaler},\n "
            f"m_mean={item(self.m_mean) * self.lateral_scaler}, m_std={item(self.m_std) * self.lateral_scaler}\n"
        )
    
    def _get_device(self):
        # look through parameters; if none exist, default to CPU
        return next(
            (p.device for p in self.parameters() if p is not None),
            torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        )


# multilayer initializer is a simple container for multiple initializers
class MultiLayerInitializer(nn.Module):
    def __init__(self, initializer_list):
        super(MultiLayerInitializer, self).__init__()
        self.initializers = nn.ModuleList(initializer_list)
        
    def __str__(self):
        st = ""
        for i, init in enumerate(self.initializers):
            st += f"Layer {i}: {init}"
        return st