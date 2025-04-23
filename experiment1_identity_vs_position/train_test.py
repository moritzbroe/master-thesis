from loss import vicreg_loss, second_derivative_loss
import torch
import torch.optim as optim
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error


# trains model using vicreg loss on a dataloader that should return pairs of images
def train_vicreg(model, ssl_loader, num_epochs=1, lr=1e-3, verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        for i, x in enumerate(ssl_loader):
            x = x.to('cuda')
            x1, x2 = x[:,0], x[:,1]
            z1, z2 = model(x1), model(x2)
            
            loss, inv,var,cov = vicreg_loss(z1, z2, cov_weight=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose:
                print('Epoch:', epoch, 'Iteration:', i, 'Loss:', loss.item(), 'Inv:', inv.item(), 'Var:', var.item(), 'Cov:', cov.item())
        

# trains model using second derivative loss on a dataloader that should return triples of images
def train_triple(model, ssl_loader, num_epochs=1, verbose=True, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        for i, x in enumerate(ssl_loader):
            x = x.to('cuda')
            x1, x2, x3 = x[:,0], x[:,1], x[:,2]
            z1, z2, z3 = model(x1), model(x2), model(x3)
            
            loss, inv, var, cov = second_derivative_loss(z1, z2, z3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose:
                print('Epoch:', epoch, 'Iteration:', i, 'Loss:', loss.item(), 'Inv:', inv.item(), 'Var:', var.item(), 'Cov:', cov.item())


# linear evaluation for extracting the position information of the single object in the image,
# i.e. trains a linear regression model on the features extracted by the model to predict the position + size
def test_position(model, supervised_loader):
    X_train, y_train = [], []
    for i, (x, y) in enumerate(supervised_loader):
        x = x.to('cuda')
        with torch.no_grad():
            z = model(x)
        X_train.append(z)
        y_train.append(y[:, 1:])

    X_train = torch.cat(X_train, dim=0).cpu().numpy()
    y_train = torch.cat(y_train, dim=0).cpu().numpy()

    X_test, y_test = [], []
    for i, (x, y) in enumerate(supervised_loader):
        x = x.to('cuda')
        with torch.no_grad():
            z = model(x)
        X_test.append(z)
        y_test.append(y[:, 1:])

    X_test = torch.cat(X_test, dim=0).cpu().numpy()
    y_test = torch.cat(y_test, dim=0).cpu().numpy()

    # train least squares regression using sklearn
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    
    pred = clf.predict(X_test)
    l2_error = mean_squared_error(y_test, pred)
    print('L2 error position extraction:', l2_error)
    return l2_error
    

# linear evaluation for extracting the class information of the single object in the image,
# i.e. trains a logistic regression model on the features extracted by the model to predict the class (triangle/square/circle)
def test_class(model, supervised_loader):
    X_train, y_train = [], []
    for i, (x, y) in enumerate(supervised_loader):
        x = x.to('cuda')
        with torch.no_grad():
            z = model(x)
        X_train.append(z)
        y_train.append(y[:, 0])

    X_train = torch.cat(X_train, dim=0).cpu().numpy()
    y_train = torch.cat(y_train, dim=0).cpu().numpy()
    y_train = y_train.round()   # need integer labels for sklearn

    X_test, y_test = [], []
    for i, (x, y) in enumerate(supervised_loader):
        x = x.to('cuda')
        with torch.no_grad():
            z = model(x)
        X_test.append(z)
        y_test.append(y[:, 0])

    X_test = torch.cat(X_test, dim=0).cpu().numpy()
    y_test = torch.cat(y_test, dim=0).cpu().numpy()
    y_test = y_test.round()    

    # train classifier using sklearn
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = (pred == y_test).mean()
    print('accuracy classification:', acc)
    return acc