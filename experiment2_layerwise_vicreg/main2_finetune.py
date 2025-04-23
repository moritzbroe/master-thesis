import numpy as np
import torch.nn as nn
import optuna
from standard_loaders import get_train_test_loader
from ssl_loader import get_basic_ssl_loader
from train_test import test_params
import matplotlib.pyplot as plt
import os
import torch.multiprocessing


# same as main.py, but now the previous tpe study is loaded and initializes the parameters of a cma-es study

torch.multiprocessing.set_sharing_strategy('file_system')

device = 'cuda'

def get_initial_params():
    # average the best 5 trial's parameters from the previous tpe study
    filename = 'tpe_final.db'
    study_name='study'
    # plot everything available in optuna
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{filename}"
    )
    # print best 5 trials
    best_trials = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value, reverse=True)[:5]
    # average the parameters
    params = {k: [] for k in best_trials[0].params.keys()}
    for trial in best_trials:
        for k, v in trial.params.items():
            params[k].append(v)
    for k, v in params.items():
        params[k] = sum(v) / len(v)
    return params

def run_optuna_study(model_fn, loss_layers, study_name='study', epochs=1, n_trials=1000, k_cross_val=10, resolution=96, batch_size=256):
    def objective(trial):
        train_loader, _ = get_train_test_loader('stl10', batch_size=512, num_workers=1, resolution=resolution)
        ssl_loader = get_basic_ssl_loader('stl10', num_workers=4, batch_size=batch_size)
        
        model = model_fn().to(device)
        params_config = []
        for i in range(len(loss_layers)):
            min_size = trial.suggest_float(f"layer_{i}_min_size", 0.05, 1.05)
            min_size = min(1.0, min_size)
            color_jitter_strength = trial.suggest_float(f"layer_{i}_color_jitter_strength", -0.05, 1.0)
            color_jitter_strength = max(0.0, color_jitter_strength)
            flip = trial.suggest_float(f"layer_{i}_flip", -0.05, 0.5)
            flip = max(0.0, flip)
            inv_coeff = trial.suggest_float(f"layer_{i}_inv_coeff", -5.0, 50.0)
            inv_coeff = max(0.0, inv_coeff)
            lr_log = trial.suggest_float(f"layer_{i}_lr_log", -4.0, -2.0)
            lr = 10**lr_log

            params_config.append([min_size, color_jitter_strength, flip, lr, inv_coeff])

        C_log = trial.suggest_float("C", -4.0, -0.0)
        C = 10**C_log

        test_acc = test_params(model, loss_layers, params_config, verbose=False, device=device, epochs=epochs, clip_grads=5.0, C=C, train_loader=train_loader, ssl_loader=ssl_loader, k_cross_val=k_cross_val, resolution=resolution)
        return test_acc

    # use the initial parameters from the previous tpe study and a small initial variance sigma0
    sampler = optuna.samplers.CmaEsSampler(x0=get_initial_params(),
                                           sigma0=0.05)

    study = optuna.create_study(direction="maximize",
                                study_name=study_name,
                                storage="sqlite:///study_cma_finetune.db",
                                load_if_exists=True,
                                sampler=sampler,
                                )
    
    study.optimize(objective, n_trials=n_trials, n_jobs=1,)


if __name__ == '__main__':
    def model_fn():
        model =  nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        for layer in model:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out')
                nn.init.zeros_(layer.bias)

        return model

    loss_layers = [2, 5, 7, 10, 12, 15, 17, 19]

    run_optuna_study(model_fn, loss_layers, epochs=1, n_trials=2000, k_cross_val=10, resolution=96, batch_size=128)
