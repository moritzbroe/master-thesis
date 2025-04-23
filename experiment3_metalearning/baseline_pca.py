import torch
from torchvision import datasets, transforms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from cuml.linear_model import LogisticRegression
 
# sets baseline by using pca as feature extractor for different k values

device = 'cuda'

def load_mnist_data(do_normalize=True):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    x_train_all = train_dataset.data.float() / 255.0  # shape [60000, 28, 28]
    y_train_all = train_dataset.targets
    x_test_all  = test_dataset.data.float() / 255.0   # shape [10000, 28, 28]
    y_test_all  = test_dataset.targets
    
    x_train_all = x_train_all.view(-1, 784)
    x_test_all  = x_test_all.view(-1, 784)
    
    # optionally normalize (mean 0, std 1)
    if do_normalize:
        scaler = StandardScaler()
        x_train_np = scaler.fit_transform(x_train_all.numpy())
        x_test_np  = scaler.transform(x_test_all.numpy())
        x_train_all = torch.from_numpy(x_train_np).float()
        x_test_all  = torch.from_numpy(x_test_np).float()
    
    y_train_all = y_train_all.long()
    y_test_all  = y_test_all.long()
    
    return x_train_all, y_train_all, x_test_all, y_test_all


def compute_pca_features(x_train_np, x_test_np, k):
    pca = PCA(n_components=k)
    x_train_pca = pca.fit_transform(x_train_np)
    x_test_pca  = pca.transform(x_test_np)
    return x_train_pca, x_test_pca


def train_linear_classifier_cuml(x_train, y_train, x_test, y_test, max_iter=100):
    x_train_np = x_train.cpu().numpy() if hasattr(x_train, "cpu") else x_train
    y_train_np = y_train.cpu().numpy() if hasattr(y_train, "cpu") else y_train
    x_test_np  = x_test.cpu().numpy() if hasattr(x_test, "cpu") else x_test
    y_test_np  = y_test.cpu().numpy() if hasattr(y_test, "cpu") else y_test


    C_val = 1e12 # corresponds to basically no regularization, not necessary for small dimensions
    model = LogisticRegression(C=C_val, max_iter=max_iter)
    model.fit(x_train_np, y_train_np)
    
    # Compute predictions and accuracies
    train_pred = model.predict(x_train_np)
    test_pred  = model.predict(x_test_np)
    train_acc = np.mean(train_pred == y_train_np)
    test_acc  = np.mean(test_pred  == y_test_np)
    print(f"Train Accuracy: {train_acc*100:.2f}%, Test Accuracy: {test_acc*100:.2f}%")
    return train_acc, test_acc, model


def run_pca_experiment(x_train, y_train, x_test, y_test, device):
    pca_k_values = list(range(1, 785, 1))
    test_accs = []
    for k in pca_k_values:
        print(f"\nPCA with k={k}")
        x_train_np = x_train.cpu().numpy()
        x_test_np  = x_test.cpu().numpy()
        x_train_pca_np, x_test_pca_np = compute_pca_features(x_train_np, x_test_np, k)
        x_train_pca_t = torch.from_numpy(x_train_pca_np).float().to(device)
        x_test_pca_t  = torch.from_numpy(x_test_pca_np).float().to(device)

        # train cuML logistic regression and print accuracies
        _, _test_acc, _ = train_linear_classifier_cuml(
            x_train_pca_t,
            y_train,
            x_test_pca_t,
            y_test,
            max_iter=3000,
        )
        test_accs.append(_test_acc)
    print(f"Final test accuracies: {test_accs}")


x_train_all, y_train_all, x_test_all, y_test_all = load_mnist_data(do_normalize=True)
x_train_all, y_train_all = x_train_all.to(device), y_train_all.to(device)
x_test_all,  y_test_all  = x_test_all.to(device),  y_test_all.to(device)


# run the experiment
run_pca_experiment(x_train_all, y_train_all, x_test_all, y_test_all, device)

exit()


# results from 5 runs, averaged and plotted:
pca_k_values = list(range(1, 33, 1)) + list(range(40, 100, 10)) + list(range(100, 784, 100)) + [784]
pca_results = [
    [0.28929999470710754, 0.3384999930858612, 0.4803999960422516, 0.6101999878883362, 0.6754999756813049, 0.7366999983787537, 0.7357999682426453, 0.7665999531745911, 0.7816999554634094, 0.802299976348877, 0.8112999796867371, 0.8172999620437622, 0.8299999833106995, 0.8356999754905701, 0.8485999703407288, 0.8533999919891357, 0.8567000031471252, 0.8569999933242798, 0.8671999573707581, 0.8689999580383301, 0.8707000017166138, 0.8700000047683716, 0.8764999508857727, 0.8787999749183655, 0.8819999694824219, 0.8836999535560608, 0.8842999935150146, 0.8888999819755554, 0.8923999667167664, 0.8930999636650085, 0.8930999636650085, 0.8944999575614929, 0.8991000056266785, 0.9016000032424927, 0.9055999517440796, 0.9072999954223633, 0.9111999869346619, 0.9126999974250793, 0.9139999747276306, 0.921999990940094, 0.9249999523162842, 0.9238999485969543, 0.9246999621391296, 0.924299955368042, 0.9241999983787537, 0.9253000020980835],
    [0.281499981880188, 0.334199994802475, 0.4940999746322632, 0.6182999610900879, 0.6753000020980835, 0.7322999835014343, 0.7394999861717224, 0.7652999758720398, 0.7835999727249146, 0.798799991607666, 0.8104000091552734, 0.8166999816894531, 0.833299994468689, 0.8379999995231628, 0.8502999544143677, 0.8557999730110168, 0.8572999835014343, 0.8592999577522278, 0.8671999573707581, 0.8700999617576599, 0.8704999685287476, 0.8708999752998352, 0.8748999834060669, 0.8787999749183655, 0.8809999823570251, 0.8833999633789062, 0.8860999941825867, 0.8885999917984009, 0.8899999856948853, 0.8914999961853027, 0.8938999772071838, 0.8916999697685242, 0.8988999724388123, 0.9018999934196472, 0.904699981212616, 0.9079999923706055, 0.9122999906539917, 0.9131999611854553, 0.9143999814987183, 0.9229999780654907, 0.9239999651908875, 0.924299955368042, 0.9241999983787537, 0.9243999719619751, 0.9233999848365784, 0.9232999682426453],
    [0.2962999939918518, 0.34209999442100525, 0.47509998083114624, 0.6128999590873718, 0.6761999726295471, 0.7364999651908875, 0.7402999997138977, 0.7676999568939209, 0.7824999690055847, 0.8008999824523926, 0.8122999668121338, 0.8181999921798706, 0.8327999711036682, 0.8371999859809875, 0.8468999862670898, 0.8531999588012695, 0.8567000031471252, 0.8572999835014343, 0.8661999702453613, 0.8689000010490417, 0.8721999526023865, 0.8718000054359436, 0.87909996509552, 0.8792999982833862, 0.87909996509552, 0.8840999603271484, 0.8855999708175659, 0.8886999487876892, 0.8922999501228333, 0.8928999900817871, 0.8926999568939209, 0.8955000042915344, 0.8976999521255493, 0.9023000001907349, 0.9054999947547913, 0.90829998254776, 0.9109999537467957, 0.9124999642372131, 0.9150999784469604, 0.9210000038146973, 0.9259999990463257, 0.9239999651908875, 0.9244999885559082, 0.9258999824523926, 0.9246000051498413, 0.9247999787330627],
    [0.2987000048160553, 0.3490999937057495, 0.48589998483657837, 0.6097999811172485, 0.676099956035614, 0.7285999655723572, 0.7411999702453613, 0.7684999704360962, 0.7846999764442444, 0.8029999732971191, 0.811199963092804, 0.8172000050544739, 0.833299994468689, 0.8353999853134155, 0.8483999967575073, 0.8542999625205994, 0.8570999503135681, 0.8578000068664551, 0.8648999929428101, 0.8695999979972839, 0.8708999752998352, 0.8715999722480774, 0.8758999705314636, 0.8783999681472778, 0.8811999559402466, 0.8836999535560608, 0.8865999579429626, 0.8884999752044678, 0.8910999894142151, 0.8917999863624573, 0.8923999667167664, 0.8923999667167664, 0.8965999484062195, 0.9027000069618225, 0.9050999879837036, 0.9095999598503113, 0.9120999574661255, 0.911300003528595, 0.9138000011444092, 0.921999990940094, 0.9236999750137329, 0.9246999621391296, 0.9239999651908875, 0.9257999658584595, 0.9248999953269958, 0.9236999750137329],
    [0.29739999771118164, 0.33739998936653137, 0.468999981880188, 0.6123999953269958, 0.675000011920929, 0.7389999628067017, 0.7372999787330627, 0.7627999782562256, 0.7839999794960022, 0.8021000027656555, 0.8098999857902527, 0.8174999952316284, 0.8324999809265137, 0.8355000019073486, 0.8452000021934509, 0.8547999858856201, 0.858199954032898, 0.8578999638557434, 0.8664000034332275, 0.8691999912261963, 0.8716999888420105, 0.8720999956130981, 0.8777999877929688, 0.8801999688148499, 0.8822000026702881, 0.8833999633789062, 0.8854999542236328, 0.8888999819755554, 0.8887999653816223, 0.8915999531745911, 0.8913999795913696, 0.8949999809265137, 0.8969999551773071, 0.902899980545044, 0.9042999744415283, 0.9071999788284302, 0.9114999771118164, 0.9132999777793884, 0.9151999950408936, 0.9222999811172485, 0.9247999787330627, 0.9243999719619751, 0.9253000020980835, 0.9236999750137329, 0.9253999590873718, 0.9251999855041504]
]
import matplotlib.pyplot as plt 

# pca_k_values = pca_k_values[:32]
results_array = np.array(pca_results)

selected_k = [32, 100, 784]
# Compute the mean, min, and max for each k value
mean_values = results_array.mean(axis=0)
min_values = results_array.min(axis=0)
max_values = results_array.max(axis=0)
# Plot the mean result as a line
# Plot the mean result as a line
plt.plot(pca_k_values, mean_values, label='Mean')

plt.show()
