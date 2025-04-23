import torch.nn as nn
import optuna
from standard_loaders import get_train_test_loader
from ssl_loader import get_basic_ssl_loader
from train_test import test_params
import torch.multiprocessing


# This code runs an optuna study using the tpe sampler to optimize the augmentation hyperparameters for a model

torch.multiprocessing.set_sharing_strategy('file_system')

device = 'cuda'

def run_optuna_study(model_fn, loss_layers, study_name='study', epochs=1, n_trials=1000, k_cross_val=10, resolution=96, batch_size=256):
    def objective(trial):
        train_loader, _ = get_train_test_loader('stl10', batch_size=512, num_workers=1, resolution=resolution)
        ssl_loader = get_basic_ssl_loader('stl10', num_workers=4, batch_size=batch_size)
        
        model = model_fn().to(device)
        params_config = []
        for i in range(len(loss_layers)):
            # some of the values are allowed to go slightly beyond their limit and are clamped back to allow better exploration of the boundary. don't know if this is actually a good idea...
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

        # get accuracy, but provide only train_loader so that the test set is not used and k-fold cross validation on the train set is used instead
        test_acc = test_params(model, loss_layers, params_config, verbose=False, device=device, epochs=epochs, clip_grads=5.0, C=C, train_loader=train_loader, ssl_loader=ssl_loader, k_cross_val=k_cross_val, resolution=resolution)
        return test_acc

    sampler = optuna.samplers.TPESampler(constant_liar=True, multivariate=True)

    study = optuna.create_study(direction="maximize",
                                study_name=study_name,
                                storage="sqlite:///study_tpe.db",
                                load_if_exists=True,
                                sampler=sampler,
                                )
    study.optimize(objective, n_trials=n_trials, n_jobs=3,)


if __name__ == '__main__':
    # define the model as a function without arguments that returns the model. has to be a nn.sequential model as its layers are later accessed via model[i:j] etc
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

    # define the layers after which the ssl loss is applied. e.g. for this model, layers 0,1,2 are one block and layers 3,4,5 are another block etc. 
    # the vicreg loss is applied after each block and updates the block's parameters (in this case only the convolutional layer's parameters).
    # each block will be trained with its own set of augmentation parameters, see train_test.py for details
    loss_layers = [2, 5, 7, 10, 12, 15, 17, 19]

    run_optuna_study(model_fn, loss_layers, epochs=1, n_trials=2000, k_cross_val=10, resolution=96, batch_size=128)
