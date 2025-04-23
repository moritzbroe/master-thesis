'''
defines fully local learning rules, that update weights/biases/lateral inhibition parameters based on inputs, 
outputs and possibly pre-activations of the layers
'''

import torch
from itertools import product
from models.models_fully_local import MLPLayer, Conv2DLayer, LocalSequential
import torch.nn as nn
import torch.nn.functional as F


# functions to generate lists of tuples corresponding to exponents used in polyonimal fully local learning rules
def generate_exponents_total_degree(num_vars, degree):
    '''
    Generate all combinations of exponents for `num_vars` variables up to `degree`, where e.g. deg(xy) = 2.
    '''
    return [exponents for exponents in product(range(degree + 1), repeat=num_vars) 
            if sum(exponents) <= degree]


def generate_exponents_partial_degree(num_vars, degree):
    '''
    Generate all combinations of exponents for `num_vars` variables up to `degree`, where deg is the degree per variable,
    i.e. deg(x y) = 1, deg(x^2 y) = 2 etc.
    '''
    return list(product(range(degree + 1), repeat=num_vars))


def generate_exponents_both_degrees(num_vars, degree_total, degree_partial):
    '''
    Generate all combinations of exponents for `num_vars` variables up to `degree_total` and `degree_partial`.
    '''
    return [exponents for exponents in product(range(degree_partial + 1), repeat=num_vars) 
            if sum(exponents) <= degree_total]


def _get_bound_fn(bound):
    if bound is None:
        return lambda x: x
    else:
        bound = bound.split(' ')
    if bound[0] == 'tanh':
        if len(bound) == 1:
            return torch.tanh
        else:
            bnd = float(bound[1])
            return lambda x: torch.tanh(x / bnd) * bnd
    elif bound[0] == 'clamp':
        if len(bound) == 1:
            return lambda x: torch.clamp(x, min=-1.0, max=1.0)
        else:
            return lambda x: torch.clamp(x, min=-float(bound[1]), max=float(bound[1]))
    else:
        raise ValueError('Unknown bound type')


# general skeleton. learning rule is a module that takes a model, its parameters and a batch of inputs and returns updated parameterss
class LearningRule(nn.Module):
    def __init__(self):
        super(LearningRule, self).__init__()
    
    def update(self, model, params, x):
        raise NotImplementedError
    
# this learning rule applies a polynomial update rule to the forward weights, biases and lateral weights of a layer or LocalSequential model. the update to e.g. w_ij will be a polynomial in x_j, s_j, y_j and w_ij
class PolynomialLearningRule(LearningRule):
    def __init__(self, exponents_w=None, exponents_b=None, exponents_m=None, lat_inh=False, bound=None, l2_reg=0.0, m_diagonal_component=True, forward_weight_bound=None):
        super(PolynomialLearningRule, self).__init__()
        self.lat_inh = lat_inh
        self.m_diagonal_component = m_diagonal_component

        self.exponents_w = exponents_w if isinstance(exponents_w, list) else generate_exponents_total_degree(4, 3)
        self.params_w = nn.Parameter(torch.zeros((len(self.exponents_w),), requires_grad=True))

        self.exponents_b = exponents_b if isinstance(exponents_b, list) else generate_exponents_total_degree(2, 3)
        self.params_b = nn.Parameter(torch.zeros((len(self.exponents_b),), requires_grad=True))

        if lat_inh:
            self.exponents_m = exponents_m if isinstance(exponents_m, list) else generate_exponents_total_degree(3, 3)
            num_par = len(self.exponents_m) if not m_diagonal_component else len(self.exponents_m) + 1
            self.params_m = nn.Parameter(torch.zeros((num_par,), requires_grad=True))
        
        self.make_unique_exponents()
        self.bound_fn = _get_bound_fn(bound)
        self.l2_reg = l2_reg
        self.forward_weight_bound = forward_weight_bound

    def make_unique_exponents(self):
        # calculates which exponents are required for each term in each update, so that later the exponentiations of each term are only performed once
        self.w_unique_exponents = [{exp[i] for exp in self.exponents_w} for i in range(4)]
        self.b_unique_exponents = [{exp[i] for exp in self.exponents_b} for i in range(2)]
        if self.lat_inh:
            self.m_unique_exponents = [{exp[0] for exp in self.exponents_m}.union({exp[1] for exp in self.exponents_m}), {exp[2] for exp in self.exponents_m}]

    def update(self, model, params, x, return_y=False):
        # for single layer, params is a dict and a dict with new params will be returned
        # can optionally return the outputs of the layer in addition to the updated parameters
        if isinstance(model, (MLPLayer, Conv2DLayer)):
            s, y = model(x, params, return_sum=True)
            delta_w = self.dw(x, s, y, params['w'])
            delta_b = self.db(y, params['b'])
            new_layer_params = {'w': params['w'] + delta_w, 'b': params['b'] + delta_b}
            if self.forward_weight_bound is not None:
                new_layer_params['w'] = new_layer_params['w'].clamp(max=self.forward_weight_bound, min=-self.forward_weight_bound)
            if self.lat_inh:
                delta_m = self.dm(y, params['m'])
                new_layer_params['m'] = params['m'] + delta_m
                if model.no_anti_inhibition:
                    new_layer_params['m'] = new_layer_params['m'].clamp(max=0)
                if model.no_self_inhibition:
                    new_layer_params['m'] = new_layer_params['m'] - torch.diag(torch.diag(new_layer_params['m']))
            return new_layer_params if not return_y else (new_layer_params, y)
        # for LocalSequential, params is a list of dicts and a list of dicts with new params will be returned
        elif isinstance(model, LocalSequential):
            new_params = []
            for layer, layer_params in zip(model.layers, params):
                if isinstance(layer, (MLPLayer, Conv2DLayer)):
                    new_layer_params, y = self.update(layer, layer_params, x, return_y=True)
                    new_params.append(new_layer_params)
                    x = y
                else:
                    x = layer(x) 
                    new_params.append(dict())
            return new_params if not return_y else (new_params, y)
        else:
            raise ValueError('Unknown model type')
        
    # updates to forward weights
    def dw(self, x, s, y, w):
        powers_x = {exp: x**exp for exp in self.w_unique_exponents[0]}
        powers_s = {exp: s**exp for exp in self.w_unique_exponents[1]}
        powers_y = {exp: y**exp for exp in self.w_unique_exponents[2]}
        powers_w = {exp: w**exp for exp in self.w_unique_exponents[3]}
        batch_size = x.shape[0]
        delta_w = torch.zeros_like(w)
        if x.dim() == 2: # MLP layer
            for exps, coeff in zip(self.exponents_w, self.params_w):
                delta_w = delta_w + coeff / batch_size * torch.mm(powers_y[exps[2]].t() * powers_s[exps[1]].t(), powers_x[exps[0]]) * powers_w[exps[3]]
        elif x.dim() == 4: # Conv2d layer
            raise NotImplementedError
        l2_reg = - self.l2_reg * w
        return self.bound_fn(delta_w) + l2_reg
    
    # updates to biases
    def db(self, y, b):
        powers_y = {exp: y**exp for exp in self.b_unique_exponents[0]}
        powers_b = {exp: b**exp for exp in self.b_unique_exponents[1]}
        batch_size = y.shape[0]
        delta_b = torch.zeros_like(b)
        if y.dim() == 2:
            for exps, coeff in zip(self.exponents_b, self.params_b):
                delta_b = delta_b + coeff / batch_size * torch.sum(powers_y[exps[0]], dim=0) * powers_b[exps[1]]
        elif y.dim() == 4:
            raise NotImplementedError
        l2_reg = - self.l2_reg * b
        return self.bound_fn(delta_b) + l2_reg

    # updates to lateral inhibition weights
    def dm(self, y, m):
        powers_y = {exp: y**exp for exp in self.m_unique_exponents[0]}
        powers_m = {exp: m**exp for exp in self.m_unique_exponents[1]}
        batch_size = y.shape[0] # todo: y[0].shape[0] before??
        delta_m = torch.zeros_like(m)
        if y.dim() == 2:
            for exps, coeff in zip(self.exponents_m, self.params_m):
                delta_m = delta_m + coeff / batch_size * torch.mm(powers_y[exps[1]].t(), powers_y[exps[0]]) * powers_m[exps[2]]
        elif y.dim() == 4:
            raise NotImplementedError
        if self.m_diagonal_component:
            delta_m = delta_m + self.params_m[-1] * torch.eye(m.shape[0]).to(m.device)
        l2_reg = - self.l2_reg * m
        return self.bound_fn(delta_m) + l2_reg
    
    # sparsifies learning rule by removing terms whose sign switched. can e.g. be called after each metastep, then all update terms whose coefficients hover around zero are eventually pruned
    def sparsify(self):
        changed = False
        print('Sparsifying...')
        if not hasattr(self, 'signs_w'):
            print('No signs found, initializing...')
            self.signs_w = torch.sign(self.params_w)
            self.signs_b = torch.sign(self.params_b)
            if self.lat_inh:
                self.signs_m = torch.sign(self.params_m)
        else:
            keep_w = torch.sign(self.params_w) == self.signs_w
            if not torch.all(keep_w):
                self.params_w = nn.Parameter(self.params_w[keep_w])
                self.exponents_w = [exp for i, exp in enumerate(self.exponents_w) if keep_w[i]]
                self.signs_w = torch.sign(self.params_w)
                changed = True

            keep_b = torch.sign(self.params_b) == self.signs_b
            if not torch.all(keep_b):
                self.params_b = nn.Parameter(self.params_b[keep_b])
                self.exponents_b = [exp for i, exp in enumerate(self.exponents_b) if keep_b[i]]
                self.signs_b = torch.sign(self.params_b)
                changed = True

            if self.lat_inh:
                keep_m = torch.sign(self.params_m) == self.signs_m
                if not torch.all(keep_m):
                    self.params_m = nn.Parameter(self.params_m[keep_m])
                    self.exponents_m = [exp for i, exp in enumerate(self.exponents_m) if keep_m[i]]
                    self.signs_m = torch.sign(self.params_m)
                    changed = True
        self.make_unique_exponents()
        return changed
    
# similar as learning rule above, but now a batch of tuples (typically augmented images) is passed to the update function.
class PolynomialLearningRulePair(LearningRule):
    def __init__(self, exponents_w=None, exponents_b=None, exponents_m=None, lat_inh=False, bound=None, l2_reg=0.0, m_diagonal_component=True, forward_weight_bound=None):
        super(PolynomialLearningRulePair, self).__init__()
        self.lat_inh = lat_inh
        self.exponents_w = exponents_w if isinstance(exponents_w, list) else generate_exponents_total_degree(7, 3)
        self.params_w = nn.Parameter(torch.zeros((len(self.exponents_w),), requires_grad=True))

        self.exponents_b = exponents_b if isinstance(exponents_b, list) else generate_exponents_total_degree(3, 3)
        self.params_b = nn.Parameter(torch.zeros((len(self.exponents_b),), requires_grad=True))

        assert len(self.exponents_w[0]) == 7
        assert len(self.exponents_b[0]) == 3

        if lat_inh:
            self.exponents_m = exponents_m if isinstance(exponents_m, list) else generate_exponents_total_degree(5, 3)
            num_par = len(self.exponents_m) if not m_diagonal_component else len(self.exponents_m) + 1
            self.params_m = nn.Parameter(torch.zeros((num_par,), requires_grad=True))
            assert len(self.exponents_m[0]) == 5

        self.make_unique_exponents()
        self.bound_fn = _get_bound_fn(bound)
        self.l2_reg = l2_reg
        self.forward_weight_bound = forward_weight_bound

    def make_unique_exponents(self):
        print('Making unique exponents...')
        self.w_unique_exponents = [{exp[i] for exp in self.exponents_w} for i in range(7)]
        self.b_unique_exponents = [{exp[i] for exp in self.exponents_b} for i in range(3)]
        if self.lat_inh:
            self.m_unique_exponents = [{exp[0] for exp in self.exponents_m}.union({exp[1] for exp in self.exponents_m}), {exp[2] for exp in self.exponents_m}.union({exp[3] for exp in self.exponents_m}), {exp[4] for exp in self.exponents_m}]


    def update(self, model, params, x, return_y=False):
        # split the input x, which is a list of two batches, corresponding to a batch of pairs
        x1, x2 = x
        if isinstance(model, (MLPLayer, Conv2DLayer)):
            s1, y1 = model(x1, params, return_sum=True)
            s2, y2 = model(x2, params, return_sum=True)
            delta_w = self.dw(x1, s1, y1, x2, s2, y2, params['w'])
            delta_b = self.db(y1, y2, params['b'])
            new_layer_params = {'w': params['w'] + delta_w, 'b': params['b'] + delta_b}
            if self.forward_weight_bound is not None:
                new_layer_params['w'] = new_layer_params['w'].clamp(max=self.forward_weight_bound, min=-self.forward_weight_bound)
            if self.lat_inh:
                delta_m = self.dm(y1, y2, params['m'])
                new_layer_params['m'] = params['m'] + delta_m
                if model.no_anti_inhibition:
                    new_layer_params['m'] = new_layer_params['m'].clamp(max=0)
                if model.no_self_inhibition:
                    new_layer_params['m'] = new_layer_params['m'] - torch.diag(torch.diag(new_layer_params['m']))
            return new_layer_params if not return_y else (new_layer_params, (y1, y2))
        elif isinstance(model, LocalSequential):
            new_params = []
            for layer, layer_params in zip(model.layers, params):
                if isinstance(layer, (MLPLayer, Conv2DLayer)):
                    new_layer_params, y = self.update(layer, layer_params, x, return_y=True)
                    new_params.append(new_layer_params)
                    x = y
                else:
                    x = layer(x) 
                    new_params.append(dict())
            return new_params if not return_y else (new_params, y)
        else:
            raise ValueError('Unknown model type')
        
    def dw(self, x1, s1, y1, x2, s2, y2, w):
        powers_x1 = {exp: x1**exp for exp in self.w_unique_exponents[0]}
        powers_s1 = {exp: s1**exp for exp in self.w_unique_exponents[1]}
        powers_y1 = {exp: y1**exp for exp in self.w_unique_exponents[2]}
        powers_x2 = {exp: x2**exp for exp in self.w_unique_exponents[3]}
        powers_s2 = {exp: s2**exp for exp in self.w_unique_exponents[4]}
        powers_y2 = {exp: y2**exp for exp in self.w_unique_exponents[5]}
        powers_w = {exp: w**exp for exp in self.w_unique_exponents[6]}
        batch_size = x1.shape[0]
        delta_w = torch.zeros_like(w)
        if x1.dim() == 2: # MLP layer
            for exps, coeff in zip(self.exponents_w, self.params_w):
                y_term = powers_s1[exps[1]] * powers_y1[exps[2]] * powers_s2[exps[4]] * powers_y2[exps[5]]
                x_term = powers_x1[exps[0]] * powers_x2[exps[3]]
                delta_w = delta_w + coeff / batch_size * torch.mm(y_term.t(), x_term) * powers_w[exps[6]]
        elif x1.dim() == 4: # Conv2d layer
            raise NotImplementedError
        l2_reg = - self.l2_reg * w
        return self.bound_fn(delta_w) + l2_reg
    
    def db(self, y1, y2, b):
        powers_y1 = {exp: y1**exp for exp in self.b_unique_exponents[0]}
        powers_y2 = {exp: y2**exp for exp in self.b_unique_exponents[1]}
        powers_b = {exp: b**exp for exp in self.b_unique_exponents[2]}
        batch_size = y1.shape[0]
        delta_b = torch.zeros_like(b)
        if y1.dim() == 2:
            for exps, coeff in zip(self.exponents_b, self.params_b):
                delta_b = delta_b + coeff / batch_size * torch.sum(powers_y1[exps[0]] * powers_y2[exps[1]], dim=0) * powers_b[exps[2]]
        elif y1.dim() == 4:
            raise NotImplementedError
        l2_reg = - self.l2_reg * b
        return self.bound_fn(delta_b) + l2_reg

    def dm(self, y1, y2, m):
        powers_y1 = {exp: y1**exp for exp in self.m_unique_exponents[0]}
        powers_y2 = {exp: y2**exp for exp in self.m_unique_exponents[1]}
        powers_m = {exp: m**exp for exp in self.m_unique_exponents[2]}
        batch_size = y1.shape[0]
        delta_m = torch.zeros_like(m)
        if y1.dim() == 2:
            for exps, coeff in zip(self.exponents_m, self.params_m):
                term = coeff / batch_size * torch.mm(powers_y1[exps[1]].t() * powers_y2[exps[3]].t(), powers_y1[exps[0]] * powers_y2[exps[2]]) * powers_m[exps[4]]
                delta_m = delta_m + term
            delta_m = delta_m + self.params_m[-1] * torch.eye(m.shape[0]).to(m.device)
        elif y1.dim() == 4:
            raise NotImplementedError
        l2_reg = - self.l2_reg * m
        return self.bound_fn(delta_m) + l2_reg
    
    def sparsify(self):
        changed = False
        print('Sparsifying...')
        if not hasattr(self, 'signs_w'):
            print('No signs found, initializing...')
            self.signs_w = torch.sign(self.params_w)
            self.signs_b = torch.sign(self.params_b)
            if self.lat_inh:
                self.signs_m = torch.sign(self.params_m)
        else:
            keep_w = torch.sign(self.params_w) == self.signs_w
            if not torch.all(keep_w):
                self.params_w = nn.Parameter(self.params_w[keep_w])
                self.exponents_w = [exp for i, exp in enumerate(self.exponents_w) if keep_w[i]]
                self.signs_w = torch.sign(self.params_w)
                changed = True

            keep_b = torch.sign(self.params_b) == self.signs_b
            if not torch.all(keep_w):
                self.params_b = nn.Parameter(self.params_b[keep_b])
                self.exponents_b = [exp for i, exp in enumerate(self.exponents_b) if keep_b[i]]
                self.signs_b = torch.sign(self.params_b)
                changed = True

            if self.lat_inh:
                keep_m = torch.sign(self.params_m) == self.signs_m
                if not torch.all(keep_m):
                    self.params_m = nn.Parameter(self.params_m[keep_m])
                    self.exponents_m = [exp for i, exp in enumerate(self.exponents_m) if keep_m[i]]
                    self.signs_m = torch.sign(self.params_m)
                    changed = True
        self.make_unique_exponents()
        return changed


# multilayer update rule has a list of learning rules, one for each layer. the update function calls the update function of each learning rule with the corresponding layer and correct inputs (which are the outputs of the previous layer).
class MultiLayerLearningRule(LearningRule):
    def __init__(self, learning_rules):
        super(MultiLayerLearningRule, self).__init__()
        self.learning_rules = nn.ModuleList(learning_rules)
        self.num_layers = len(learning_rules)

    def update(self, model, params, x):
        assert isinstance(model, LocalSequential), 'model must be a LocalSequential'
        i = 0
        new_params = []
        for layer, layer_params in zip(model.layers, params):
            if isinstance(layer, (MLPLayer, Conv2DLayer)):
                new_layer_params, y = self.learning_rules[i].update(layer, layer_params, x, return_y=True)
                new_params.append(new_layer_params)
                x = y
                i += 1
            else:
                if isinstance(x, (list, tuple)):
                    x = [layer(z) for z in x]
                else:
                    x = layer(x) 
                new_params.append(dict())
        assert i == self.num_layers, 'Not all update rules used'
        return new_params
    
    def sparsify(self):
        changed = False
        for update_rule in self.learning_rules:
            if update_rule.sparsify():
                changed = True
        return changed

    def __str__(self):
        return '\n'.join([str(update_rule) for update_rule in self.learning_rules])

