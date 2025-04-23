
from data.mnist_datasets import get_train_test_loader, get_ssl_loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression


# sets a baseline by training vicreg with the same augmentations as before end-to-end on a simple mlp

def extract_features(loader, model, device='cuda'):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            feat = model(images)
            if len(feat.shape) > 2:
                raise ValueError("The model output should be a feature representation.")
            features.append(feat)
            labels.append(targets)
    return torch.cat(features, dim=0).cpu(), torch.cat(labels, dim=0).cpu()


def linear_probe(
        model, 
        train_loader, 
        test_loader, 
        device='cuda',
        C=0.01,
        verbose=True,
    ):
    print('probe')
    # Pre-extract features for both train and test sets.
    X_train, y_train = extract_features(train_loader, model, device=device)
    X_test, y_test = extract_features(test_loader, model, device=device)
    
    # Convert tensors to NumPy arrays.
    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy()
    X_test_np = X_test.numpy()
    y_test_np = y_test.numpy()
    
    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=5000,
    )
    clf.fit(X_train_np, y_train_np)
    
    # Get predictions and compute accuracy.
    y_test_pred = clf.predict(X_test_np)
    test_acc = np.mean(y_test_pred == y_test_np)
    y_train_pred = clf.predict(X_train_np)
    train_acc = np.mean(y_train_pred == y_train_np)
    
    if verbose:
        print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    return test_acc


# define vicreg loss

def invariance_loss(z1, z2):
    return F.mse_loss(z1, z2)

def var_loss(z, eps=1e-6):
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(0.1 - std).mean()

def cov_loss(z):
    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (z.size(0) - 1)
    diag = torch.eye(cov.size(0), device=z.device)
    cov_offdiag = cov[~diag.bool()]
    return cov_offdiag.pow(2).sum() / z.size(1)
    
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

# simple mlp, 256-32
model = nn.Sequential(
    nn.Flatten(start_dim=1),
    nn.Linear(784, 256),
    nn.Sigmoid(),
    nn.Linear(256, 32),
    nn.Sigmoid(),
).to('cuda')

model[1].weight.data.normal_(0, 1.0 / np.sqrt(784))
model[3].weight.data.normal_(0, 1.0 / np.sqrt(256))

ssl_loader = get_ssl_loader(num_workers=10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
train_loader, test_loader = get_train_test_loader(normalization='none', batch_size=1024)

# train vicreg, track linear evaluation accuracy
for epoch in range(100):
    print(f'Epoch {epoch}')
    for batch in ssl_loader:
        optimizer.zero_grad()
        x1, x2 = batch
        x1 = x1.to('cuda')
        x2 = x2.to('cuda')
        out1 = model(x1)
        out2 = model(x2)
        loss, inv, var, cov = vicreg_loss(out1, out2, inv_weight=250, var_weight=25, cov_weight=1000)
        loss.backward()
        optimizer.step()
        print(f'Loss: {loss.item()}, Invariance: {inv.item()}, Variance: {var.item()}, Covariance: {cov.item()}')

    test_acc = linear_probe(model, train_loader, test_loader, device='cuda', C=1000000)
    print(f'Test Accuracy: {test_acc:.4f}')
    print("==" * 30)


# results:
# mlp 256-32: ~ 94.2% test acc
# mlp 32: ~ 91.2% test acc