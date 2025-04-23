import torch.nn as nn
from standard_loaders import get_train_test_loader
from ssl_loader import get_basic_ssl_loader
from linear_probe import linear_probe, linear_probe_cross_validated
from train_test import test_params
import optuna

'''
This script can be used to evaluate the final parameters by training and evaluating the model with them several times etc.
'''

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


# the k best parameters after cma-es finetuning are averaged to obtain the final parameters
# in this case only the single best parameters are used, i.e. k=1:
best_k = 1
study = optuna.load_study(
    study_name='study',
    storage=f"sqlite:///study_cma_finetune.db",
)
trials = [t for t in study.get_trials() if t.value is not None]
trials = sorted(trials, key=lambda t: t.value, reverse=True)[:best_k]
params_dict = {k: [] for k in trials[0].params.keys()}
for trial in trials:
    for k in params_dict.keys():
        params_dict[k].append(trial.params[k])
params_dict = {k: sum(v)/len(v) for k, v in params_dict.items()}
params = []
for i in range(len(loss_layers)):
    params.append([
        min(1.0, params_dict[f"layer_{i}_min_size"]), 
        max(0.0, params_dict[f"layer_{i}_color_jitter_strength"]), 
        max(0.0, params_dict[f"layer_{i}_flip"]), 
        10**params_dict[f"layer_{i}_lr_log"], 
        max(0.0, params_dict[f"layer_{i}_inv_coeff"])
    ])
bestC = 10**params_dict['C']

# now the parameters are in the correct format

# # these are the parameters from my run, which is saved in 'study_cma_finetune.db'. uncomment to use them directly and modify them manually to try things out.
# params = [
#     [0.9708193688750028, 0.6843295021334911, 0.022320436179468578, 0.003974402937700223, 15.953912371077603], 
#     [0.9501358606698114, 0.64848835803986, 0.07219281576777947, 0.005642004883724312, 17.965519154994226], 
#     [0.818324063209774, 0.36215825043563343, 0.16148061607062403, 0.0012419373719386177, 16.52481974271524], 
#     [0.7027325322213805, 0.7762445467571115, 0.3700361984098982, 0.0021361400303745916, 17.239684981556998], 
#     [0.3488859679824883, 0.8008049134448773, 0.2043495315791642, 0.00021940825290441202, 15.599374429156793], 
#     [0.5452954532496426, 0.8942266219888995, 0.24690399007800884, 0.00540971235162393, 25.124444030378182], 
#     [0.21289757205272897, 0.2140733613734248, 0.17541459832692308, 0.001514030710182264, 14.769340562450225], 
#     [0.8037841077052943, 0.3249487655286966, 0.3228339343657651, 0.0008376802246849007, 38.29516138987039]]
# bestC = 10**-0.883


batch_size = 128
resolution = 96
device = 'cuda'
k_cross_val = 10

train_loader, test_loader = get_train_test_loader('stl10', batch_size=512, num_workers=1, resolution=resolution)
ssl_loader = get_basic_ssl_loader('stl10', num_workers=4, batch_size=batch_size)


def test_k_times_sequential(k=1, n_epochs=1):
    # Train each layer with n_epochs epochs, i.e. first layer for n_epochs, second layer for n_epochs, etc., then evaluate on training set with k-fold cross validation and on train + test set
    # Do this k times and average the results
    val_accs = []
    test_accs = []
    for i in range(k):
        print(f'N_EPOCHS: {n_epochs}, EVALUATION: {i+1}/{k}')
        model = model_fn().to(device)
        val_acc = test_params(model, loss_layers, params, verbose=False, device=device, epochs=n_epochs, clip_grads=5.0, C=bestC, train_loader=train_loader, ssl_loader=ssl_loader, k_cross_val=k_cross_val, resolution=resolution)
        val_accs.append(val_acc)
        test_acc = linear_probe(model, train_loader, test_loader, device=device, verbose=False, C=bestC)
        test_accs.append(test_acc)
        print('val acc:', val_acc, 'test acc:', test_acc)
    print('EPOCHS:', n_epochs, 'NUM EVALUATIONS:', k)
    print('val accs: ', val_accs)
    print('test accs: ', test_accs)
    mean_val = sum(val_accs) / len(val_accs)
    mean_test = sum(test_accs) / len(test_accs)
    std_val = (sum([(x - mean_val) ** 2 for x in val_accs]) / len(val_accs)) ** 0.5
    std_test = (sum([(x - mean_test) ** 2 for x in test_accs]) / len(test_accs)) ** 0.5
    print('val_acc:', mean_val, '+-', std_val)
    print('test_acc:', mean_test, '+-', std_test)


def test_k_times_simultaneous(k=1, n_epochs=1):
    # This trains each layer for one epoch starting from the first layer, then the second layer, etc., then starts again at the first layer etc until each layer has been trained for n_epochs epochs
    # After each layer has been trained for i epochs, evaluate on training set with k-fold cross validation and on train + test set
    # Do this k times and average the results
    # This is similar to test_k_times_sequential, but the order of training is different and we get results after each epoch and not just after the final one
    val_accs = {epochs: [] for epochs in range(1, n_epochs+1)}
    test_accs = {epochs: [] for epochs in range(1, n_epochs+1)}
    for i in range(k):
        model = model_fn().to(device)
        for epoch in range(1, n_epochs+1):
            print(f'N_EPOCHS: {n_epochs}, EVALUATION: {i+1}/{k}, EPOCH: {epoch}/{n_epochs}')
            val_acc = test_params(model, loss_layers, params, verbose=False, device=device, epochs=1, clip_grads=5.0, C=bestC, train_loader=train_loader, ssl_loader=ssl_loader, k_cross_val=k_cross_val, resolution=resolution)
            val_accs[epoch].append(val_acc)
            test_acc = linear_probe(model, train_loader, test_loader, device=device, verbose=False, C=bestC)
            test_accs[epoch].append(test_acc)
            print('val acc:', val_acc, 'test acc:', test_acc)
    for epochs in range(1, n_epochs+1):
        mean_val = sum(val_accs[epochs]) / len(val_accs[epochs])
        mean_test = sum(test_accs[epochs]) / len(test_accs[epochs])
        std_val = (sum([(x - mean_val) ** 2 for x in val_accs[epochs]]) / len(val_accs[epochs])) ** 0.5
        std_test = (sum([(x - mean_test) ** 2 for x in test_accs[epochs]]) / len(test_accs[epochs])) ** 0.5
        print('EPOCHS:', epochs, 'NUM EVALUATIONS:', k)
        print('val accs: ', val_accs[epochs])
        print('test accs: ', test_accs[epochs])
        print('val_acc:', mean_val, '+-', std_val)
        print('test_acc:', mean_test, '+-', std_test)



test_k_times_sequential(k=5, n_epochs=1)
