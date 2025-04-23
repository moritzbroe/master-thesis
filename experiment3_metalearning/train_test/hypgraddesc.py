
from train_test.train import train
import matplotlib.pyplot as plt
import torch
import pprint
import numpy as np
import torch.nn.functional as F
import traceback

# central function for performing hypergradient descent to optimize a learning rule and potentially initializer
# model has to be passed as a function without arguments, i.e. model = model_fn(), so that new model can be created in every metastep
# in each metastep the following is done:
# - model is created
# - model parameters are initialized
# - model parameters are updated for some inner steps with learning_rule
# - loss function is evaluated on the final model parameters, takes as input model and params and returns loss
# - loss is backpropagated through the inner update steps to receive gradients for learning rule and potentially initializer to update their parameters (i.e. the hyperparameters)
# Several options available for gradually increasing the number of inner steps and regularizing hyperparameters etc

# Prints a lot of things like learning rule and initializer at each metastep, could use some polishing

def hypgraddesc(
        model_fn,
        learning_rule, 
        initializer,
        unsup_loader, 
        loss_fct,
        train_initializer=False,
        inner_steps=50, # number of inner steps, can also be a function of the metastep
        initial_steps=0,   # the number of inner update steps done before tracking gradients - if > 0, it will do truncated backpropagation 
        reinitialize_every_steps=1, # if this is set above 1, then the model will be reinitialized only every reinitialize_every_steps metasteps instead of each metastep
        meta_steps=1000,    # total number of meta steps, can just be set to a high value as breaking with ctrl+c will print results and show plots as well
        meta_lr=1e-3,   # learning rate for the hyperparamters (learning rule + initializer)
        lr_warmup=100,  # linear lr warmup for the hyperparameters
        clip_grads=None,    # clip gradients for the hyperparameters
        l1_reg_learning_rule=0.0,   # l1 regularization for the learning rule, can also be set as a function of the metastep
        sparsify=None,  # if set, will sparsify the learning rule after this many metasteps
        plot_hyperparams=None,  # can be set to a function of the learning rule, these hyperparameters will be plotted in the end
        device='cuda:0',
        l2_reg_learning_rule=0.0,
        accumulate_updates=False,   # if set to true and using reinitialize_every_steps, then the updates to the learning rule will be accumulated and only applied when reinitializing
        finetune_steps=0,   # number of steps to finetune the model after training the learning rule
        finetune_lr_factor=0.1,     # factor to reduce learning rate for the finetuning steps
        optimizer_name='adam',
):
    if train_initializer:
        assert reinitialize_every_steps is not None, 'if training the initializer, must reinitialize every so often'
        assert initial_steps == 0, 'if training the initializer, must not do any initial steps'
    if plot_hyperparams:
        assert sparsify is None, 'currently cannot plot hyperparams if sparsifying'
    params_lst = []

    def get_optimizer_scheduler():    
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop
        else: 
            raise ValueError(f'Unknown optimizer: {optimizer_name}')
        if train_initializer:
            optimizer = optimizer(list(learning_rule.parameters()) + list(initializer.parameters()), lr=meta_lr, weight_decay=l2_reg_learning_rule)
        else:
            optimizer = optimizer(learning_rule.parameters(), lr=meta_lr, weight_decay=l2_reg_learning_rule)
        if lr_warmup > 0:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / lr_warmup) if step < meta_steps - finetune_steps else finetune_lr_factor) 
        else:
            scheduler = None
        return optimizer, scheduler
    optimizer, scheduler = get_optimizer_scheduler()

    try:
        for metastep in range(meta_steps):
            print('_'*80)
            optim_steps_current = inner_steps if isinstance(inner_steps, int) else inner_steps(metastep)
            
            initial_steps_current = initial_steps if isinstance(initial_steps, int) else initial_steps(metastep)
            l1_reg_current = l1_reg_learning_rule if isinstance(l1_reg_learning_rule, (float, int)) else l1_reg_learning_rule(metastep)

            if (reinitialize_every_steps is not None and metastep % reinitialize_every_steps == 0) or metastep == 0:
                print('REINITIALIZING')
                model = model_fn()
                params = model.get_initial_params(initializer)
                params = [{k: v.to(device) for k, v in layer_params.items()} for layer_params in params]
                params = train(model, params, unsup_loader, learning_rule=learning_rule, steps=initial_steps_current, grad=False, device=device)
            

            print('metastep:', metastep, ', local learning steps:', optim_steps_current)
            print('learning rule:')
            pprint.pp(dict([p for p in learning_rule.named_parameters() if p[1].numel() > 0]))
            if train_initializer:
                print('initializer:')
                print(initializer)
            # print('MEMORY'  , torch.cuda.memory_allocated() / 1e9)
            params = train(model, params, unsup_loader, learning_rule=learning_rule, steps=optim_steps_current, grad=True, use_checkpointing=True, device=device)

            loss = loss_fct(model, params)

            for v in learning_rule.parameters():
                loss = loss + l1_reg_current * F.smooth_l1_loss(v, torch.zeros_like(v), beta=meta_lr/10, reduction='sum')
            # loss = loss + l1_reg_current * sum([v.abs().sum() for v in learning_rule.parameters()])

            optimizer.zero_grad()
            loss.backward()

            print('GRADS:', [p.grad.norm().item() for p in learning_rule.parameters()])

            # update learning rule
            if (not accumulate_updates) or metastep % reinitialize_every_steps == reinitialize_every_steps - 1:
                if clip_grads:
                    torch.nn.utils.clip_grad_norm_(learning_rule.parameters(), clip_grads)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                if sparsify is not None and metastep >= sparsify:
                    changed = learning_rule.sparsify()
                    if changed:
                        optimizer, scheduler = get_optimizer_scheduler()

            if plot_hyperparams:
                params_lst.append(plot_hyperparams(learning_rule).detach().cpu().numpy().copy())
            
            params = [{k: v.detach().requires_grad_() for k, v in layer_params.items()} for layer_params in params] # TODO: remove?

    except Exception as e:
        print("EXCEPTION\n", e)
        traceback.print_exc()
        pass 

    finally:
        if plot_hyperparams is not None:
            params_lst = np.array(params_lst)
            for i in range(params_lst.shape[1]):
                plt.plot(params_lst[:, i], label=str(i))
            plt.legend()
            plt.title('Hyperparameters')
            plt.show()
            
        return params_lst