import torch 
from cuml.linear_model import LogisticRegression
import torch.nn as nn 
import torch.nn.functional as F
import cupy as cp


# extract features from the model and dataloader, i.e. iterate through it once and get a large tensor of the representations 
def extract_features(loader, model, params, device='cuda'):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            feat = model(images, params)
            if len(feat.shape) > 2:
                raise ValueError("The model output should be a feature representation.")
            features.append(feat)
            labels.append(targets)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


# this uses cuML to fit a logistic regression model to the features extracted from the model with the train_loader
# one could use a pytorch classifier instead but would need to be careful to detach things at the right place - however, this is much faster for reaching convergence with smaller datasets/representation sizes. 
# then the logistic regression model is translated to a pytorch classifier and one final batch of size loss_batch_size from the 
# train loader (or test loader if use_test_loader=True) is passed through this differentiable model to compute the loss, which is attached to the model's params in the computational graph 
def linear_probe(
    model, 
    params,
    train_loader, 
    test_loader,
    grad=True,
    device='cuda',
    loss_batch_size=1024,
    use_test_loader=False,
    C=0.01,
):
    # extract features using the provided model.
    X_train, y_train = extract_features(train_loader, model, params, device=device)
    X_test, y_test = extract_features(test_loader, model, params, device=device)

    # need to do some weird conversion to get the data into the right format for cuML
    X_train = cp.fromDlpack(torch.utils.dlpack.to_dlpack(X_train.detach().to(device))).astype(cp.float32)
    y_train = cp.fromDlpack(torch.utils.dlpack.to_dlpack(y_train.detach().to(device))).astype(cp.int32)
    X_test = cp.fromDlpack(torch.utils.dlpack.to_dlpack(X_test.detach().to(device))).astype(cp.float32)
    y_test = cp.fromDlpack(torch.utils.dlpack.to_dlpack(y_test.detach().to(device))).astype(cp.int32)
    
    # fit the logistic regression using cuML (which now operates on CuPy arrays).
    clf = LogisticRegression(
        C=C,
        max_iter=5000,
    )
    clf.fit(X_train, y_train)

    # get predictions and compute accuracy, only for printing
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_acc = cp.mean(y_train_pred == y_train).item()
    test_acc = cp.mean(y_test_pred == y_test).item()

    print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    if not grad:
        return train_acc, test_acc

    # copy the classifier parameters to a torch model somehow
    clf_torch = nn.Linear(X_train.shape[1], clf.coef_.shape[0], bias=True).to(device)
    with torch.no_grad():
        weight_tensor = torch.from_dlpack(clf.coef_.toDlpack()).to(device).float()
        bias_tensor = torch.from_dlpack(clf.intercept_.toDlpack()).to(device).float()
        clf_torch.weight.copy_(weight_tensor)
        clf_torch.bias.copy_(bias_tensor)

    # now clf_torch is a standard pytorch linear classifier fully adapted to the features

    clf_torch.eval()

    # Gather a batch for computing the loss (gradients).
    inputs = []
    labels = []
    loader = test_loader if use_test_loader else train_loader
    num_batches = min(1, loss_batch_size // loader.batch_size)
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        inputs.append(x)
        labels.append(y)
        if i >= num_batches - 1:
            break

    # pass this evaluation batch through the model to get the loss, which is returend along with the accuarcies        
    inputs = torch.cat(inputs, dim=0)
    labels = torch.cat(labels, dim=0)
    inputs.requires_grad = True
    reps = model(inputs, params)

    logits = clf_torch(reps)
    loss = F.cross_entropy(logits, labels)

    return loss, train_acc, test_acc
