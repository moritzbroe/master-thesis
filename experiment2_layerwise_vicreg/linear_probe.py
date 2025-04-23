import torch
import numpy as np
from cuml.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import cupy as cp


# Code for linear probing using cuML, i.e. using logistic regression to train a linear classifier on top of the features extracted from the model
# Pre-extracts features from model to train and test the linear classifier much faster 

def extract_features(loader, model, device='cuda'):
    '''Extract features from the model for the given data loader'''
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
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)

def linear_probe(
        model, 
        train_loader, 
        test_loader, 
        device='cuda',
        C=0.01,
        verbose=True,
    ):
    '''Train using the train_loader and test using the test_loader, only used for final evaluation'''
    X_train, y_train = extract_features(train_loader, model, device=device)
    X_test, y_test = extract_features(test_loader, model, device=device)
    
    # Convert torch tensors to CuPy arrays directly using DLPack.
    X_train_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(X_train.detach().to(device))).astype(cp.float32)
    y_train_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(y_train.detach().to(device))).astype(cp.int32)
    X_test_cp  = cp.fromDlpack(torch.utils.dlpack.to_dlpack(X_test.detach().to(device))).astype(cp.float32)
    y_test_cp  = cp.fromDlpack(torch.utils.dlpack.to_dlpack(y_test.detach().to(device))).astype(cp.int32)
    
    # Fit the logistic regression using cuML.
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(X_train_cp, y_train_cp)
    
    # Get predictions and compute accuracy.
    y_test_pred = clf.predict(X_test_cp)
    test_acc = cp.mean(y_test_pred == y_test_cp).item()
    y_train_pred = clf.predict(X_train_cp)
    train_acc = cp.mean(y_train_pred == y_train_cp).item()
    
    if verbose:
        print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    return test_acc

def linear_probe_cross_validated(
        model, 
        dataloader, 
        n_folds=5, 
        device='cuda', 
        C=0.01,
        verbose=True
    ):
    '''Train using KFold cross-validation on the train set, this will be used for hyperparameter tuning'''
    # Extract features and labels from the entire dataset.
    X, y = extract_features(dataloader, model, device=device)
    
    # Convert tensors to CuPy arrays using DLPack.
    X_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(X.detach().to(device))).astype(cp.float32)
    y_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(y.detach().to(device))).astype(cp.int32)
    
    # Since KFold expects CPU (NumPy) arrays, convert CuPy arrays to NumPy.
    X_np = cp.asnumpy(X_cp)
    y_np = cp.asnumpy(y_cp)
    
    # Prepare KFold cross validation.
    kf = KFold(n_splits=n_folds, shuffle=True)
    
    fold_accuracies = []
    for fold, (train_index, test_index) in enumerate(kf.split(X_np), 1):
        if verbose:
            print('linear evaluation fold', fold)
        # Split data for this fold, converting back to CuPy arrays.
        X_train_fold = cp.asarray(X_np[train_index]).astype(cp.float32)
        X_test_fold  = cp.asarray(X_np[test_index]).astype(cp.float32)
        y_train_fold = cp.asarray(y_np[train_index]).astype(cp.int32)
        y_test_fold  = cp.asarray(y_np[test_index]).astype(cp.int32)
        
        # Initialize and train the logistic regression classifier.
        clf = LogisticRegression(C=C, max_iter=5000)
        clf.fit(X_train_fold, y_train_fold)
        
        # Predict and compute accuracy on the test fold.
        y_pred = clf.predict(X_test_fold)
        test_acc = cp.mean(y_pred == y_test_fold).item()
        fold_accuracies.append(test_acc)
        if verbose:
            print('done, acc', test_acc)
    
    # Compute and print the average accuracy.
    avg_acc = np.mean(fold_accuracies)
    if verbose:
        print(f"Average Accuracy: {avg_acc:.4f}")
    
    return float(avg_acc)



# sklearn version, uncomment if you want to use it instead and comment/delete the stuff above. Then installation of cuML is not needed.

# import torch
# import numpy as np
# from sklearn.model_selection import KFold
# from sklearn.linear_model import LogisticRegression


# def extract_features(loader, model, device='cuda'):
#     features = []
#     labels = []
#     with torch.no_grad():
#         for images, targets in loader:
#             images = images.to(device)
#             feat = model(images)
#             if len(feat.shape) > 2:
#                 raise ValueError("The model output should be a feature representation.")
#             features.append(feat)
#             labels.append(targets)
#     return torch.cat(features, dim=0).cpu(), torch.cat(labels, dim=0).cpu()


# def linear_probe(
#         model, 
#         train_loader, 
#         test_loader, 
#         device='cuda',
#         C=0.01,
#         verbose=True,
#     ):
#     print('probe')
#     # Pre-extract features for both train and test sets.
#     X_train, y_train = extract_features(train_loader, model, device=device)
#     X_test, y_test = extract_features(test_loader, model, device=device)
    
#     # Convert tensors to NumPy arrays.
#     X_train_np = X_train.numpy()
#     y_train_np = y_train.numpy()
#     X_test_np = X_test.numpy()
#     y_test_np = y_test.numpy()
    
#     clf = LogisticRegression(
#         C=C,
#         solver="lbfgs",
#         max_iter=5000,
#     )
#     clf.fit(X_train_np, y_train_np)
    
#     # Get predictions and compute accuracy.
#     y_test_pred = clf.predict(X_test_np)
#     test_acc = np.mean(y_test_pred == y_test_np)
#     y_train_pred = clf.predict(X_train_np)
#     train_acc = np.mean(y_train_pred == y_train_np)
    
#     if verbose:
#         print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
#     return test_acc


# def linear_probe_cross_validated(
#         model, 
#         dataloader, 
#         n_folds=5, 
#         device='cuda', 
#         C=0.01,
#         verbose=True
#     ):
#     # Extract features and labels from the entire dataset.
#     X, y = extract_features(dataloader, model, device=device)
    
#     # Convert tensors to NumPy arrays.
#     X_np = X.numpy()
#     y_np = y.numpy()
    
#     # Prepare KFold cross validation.
#     kf = KFold(n_splits=n_folds, shuffle=True)
    
#     fold_accuracies = []
#     for fold, (train_index, test_index) in enumerate(kf.split(X_np), 1):
#         print('linear evaluation fold', fold)
#         # Split data for this fold.
#         X_train, X_test = X_np[train_index], X_np[test_index]
#         y_train, y_test = y_np[train_index], y_np[test_index]
#         # Initialize and train the logistic regression classifier.
#         clf = LogisticRegression(
#             C=C,
#             solver="lbfgs",
#             max_iter=5000,
#         )
#         clf.fit(X_train, y_train)
        
#         # Predict and compute accuracy on the test fold.
#         y_pred = clf.predict(X_test)
#         test_acc = np.mean(y_pred == y_test)
#         fold_accuracies.append(test_acc)
#         print('done')
#         if verbose:
#             print(f"Fold {fold} Accuracy: {test_acc:.4f}")
    
#     # Compute and print the average accuracy.
#     avg_acc = np.mean(fold_accuracies)
#     if verbose:
#         print(f"Average Accuracy: {avg_acc:.4f}")
    
#     return avg_acc
