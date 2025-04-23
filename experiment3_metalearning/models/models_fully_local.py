import torch
import torch.nn.functional as F
from torch import nn
import copy 
from initializers.initializer import Initializer, MultiLayerInitializer

# this defines mlp layers and conv layers with external parameters, i.e. where the parameters 
# of each layer are a dict and passed to the layer for the forward pass. different inhibition modes, 
# including lateral inhibition with weights, can be used


def lateral_inhibition(sigma, s, m, steps, step_size):
    if s.dim() == 4:
        B, C, H, W = s.shape
        s = s.permute(0, 2, 3, 1)   # [B,H,W,D]
        s = s.reshape(-1, s.shape[-1])  # [BHW,D]
        inhibited = lateral_inhibition(sigma, s, m, steps=steps, step_size=step_size)
        return inhibited.reshape(B, H, W, C).permute(0, 3, 1, 2)
    y = sigma(s)
    for step in range(steps):
        y = y + step_size * (sigma(s + y @ m.t()) - y)
    return y


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation, inh_mode=None, no_anti_inhibition=False, 
                 no_self_inhibition=False, inhibition_steps=10, inhibition_factor=0.1):
        super(MLPLayer, self).__init__()
        self.activation = activation if activation is not None else lambda x: x
        self.inh_mode = inh_mode
        self.no_anti_inhibition = no_anti_inhibition # does not allow neurons having lateral connections to themselves
        self.no_self_inhibition = no_self_inhibition # does not allow positive lateral connections, should be implemented differently...
        self.inhibition_steps = inhibition_steps
        self.inhibition_factor = inhibition_factor
        self.in_dim = in_dim
        self.out_dim = out_dim

    def get_initial_params(self, initializer):
        w = initializer.get_forward_weights(self.out_dim, self.in_dim)
        b = initializer.get_biases(self.out_dim)
        if self.inh_mode == 'weights':
            m = initializer.get_lateral_weights(self.out_dim)
        return {'w': w, 'b': b, 'm': m} if self.inh_mode == 'weights' else {'w': w, 'b': b}

    def forward(self, x, params, return_sum=False):
        # forward method takes inputs x and params, where params is a dict, returns either activations y or pre-activations s and activations y
        w = params['w']
        b = params['b']
        if self.inh_mode == 'weights':
            m = params['m']
        s = x @ w.t() + b
        if self.inh_mode == 'weights':
            inhibited = lateral_inhibition(self.activation, s, m, 
                                           steps=self.inhibition_steps, 
                                           step_size=self.inhibition_factor)
            return (s, inhibited) if return_sum else inhibited
        y = self.activation(s)
        if self.inh_mode == 'l2':
            y = y / (1e-6 + torch.norm(y, dim=1, keepdim=True))
        elif self.inh_mode == 'l1' and not self.activation == torch.exp:
            y = y / (1e-6 + torch.norm(y, p=1, dim=1, keepdim=True))
        elif self.inh_mode is None:
            pass
        else:
            raise ValueError('Unknown inhibition mode')
        return (s, y) if return_sum else y
    

# convolutional layer, not tested and learning rules not implemented yet for this layer
class Conv2DLayer(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, 
                 kernel_size, stride=1, padding=0, 
                 activation=None, 
                 inh_mode=None, 
                 no_anti_inhibition=False, 
                 no_self_inhibition=False, 
                 inhibition_steps=10, 
                 inhibition_factor=0.1):
        super(Conv2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation if activation is not None else lambda x: x
        self.inh_mode = inh_mode
        self.no_anti_inhibition = no_anti_inhibition
        self.no_self_inhibition = no_self_inhibition
        self.inhibition_steps = inhibition_steps
        self.inhibition_factor = inhibition_factor

    def get_initial_params(self, initializer):
        w = initializer.get_forward_conv_weights(
            out_channels=self.out_channels, 
            in_channels=self.in_channels, 
            kernel_size=self.kernel_size
        )
        b = initializer.get_biases(self.out_channels)
        if self.inh_mode == 'weights':
            m = initializer.get_lateral_weights(self.out_channels)
            return {'w': w, 'b': b, 'm': m}
        else:
            return {'w': w, 'b': b}

    def forward(self, x, params, return_sum=False):
        w = params['w']
        b = params['b']
        if self.inh_mode == 'weights':
            m = params['m']

        s = F.conv2d(
            x, w, b, 
            stride=self.stride, 
            padding=self.padding, 
        )
        if self.inh_mode == 'weights':
            inhibited = lateral_inhibition(
                sigma=self.activation,
                s=s, 
                m=m,
                steps=self.inhibition_steps,
                step_size=self.inhibition_factor
            )
            return (s, inhibited) if return_sum else inhibited
        
        y = self.activation(s)

        if self.inh_mode == 'l2':
            norm_val = torch.norm(y.view(y.size(0), -1), dim=1, keepdim=True)
            norm_val = norm_val.view(-1, 1, 1, 1).clamp_min(1e-6)
            y = y / norm_val
        elif self.inh_mode == 'l1' and not self.activation == torch.exp:
            norm_val = y.view(y.size(0), -1).abs().sum(dim=1, keepdim=True)
            norm_val = norm_val.view(-1, 1, 1, 1).clamp_min(1e-6)
            y = y / norm_val
        elif self.inh_mode is None:
            pass
        else:
            raise ValueError('Unknown inhibition mode in Conv2DLayer')

        return (s, y) if return_sum else y


# this is a sequential container that takes a list of layers and applies them in order
# its forward method now takes a list of parameter dicts, one for each layer
class LocalSequential(nn.Module):
    def __init__(self, layers):
        super(LocalSequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, params):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (MLPLayer, Conv2DLayer)):
                x = layer(x, params[i])
            else:
                x = layer(x)
        return x
    
    def get_initial_params(self, initializer):
        params = []
        if isinstance(initializer, MultiLayerInitializer):
            i = 0
            for layer in self.layers:
                if isinstance(layer, (MLPLayer, Conv2DLayer)):
                    params.append(layer.get_initial_params(initializer.initializers[i]))
                    i += 1
                else:
                    params.append(dict())
        else:
            for layer in self.layers:
                if isinstance(layer, (MLPLayer, Conv2DLayer)):
                    params.append(layer.get_initial_params(initializer))
                else:
                    params.append(dict())
        return params
    

    