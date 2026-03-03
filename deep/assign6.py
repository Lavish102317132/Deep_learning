import numpy as np
import pandas as pd

np.random.seed(7)

N = 1000
X = np.random.uniform(-2, 2, (N, 2))
r = np.sqrt(X[:,0]**2 + X[:,1]**2)
y = (r < 1.0).astype(int).reshape(-1,1)

perm = np.random.permutation(N)
X = X[perm]
y = y[perm]

t1 = int(0.7*N)
t2 = int(0.85*N)

X_train, y_train = X[:t1], y[:t1]
X_val, y_val = X[t1:t2], y[t1:t2]
X_test, y_test = X[t2:], y[t2:]

def init_params(dims):
    p = {}
    for i in range(1, len(dims)):
        p["W"+str(i)] = np.random.randn(dims[i-1], dims[i]) * np.sqrt(2/dims[i-1])
        p["b"+str(i)] = np.zeros((1, dims[i]))
    return p

def tanh(z):
    return np.tanh(z)

def dtanh(z):
    return 1 - np.tanh(z)**2

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward(x, p):
    c = {}
    a = x
    L = len(p)//2
    for i in range(1, L):
        z = a @ p["W"+str(i)] + p["b"+str(i)]
        a = tanh(z)
        c["Z"+str(i)] = z
        c["A"+str(i)] = a
    zL = a @ p["W"+str(L)] + p["b"+str(L)]
    aL = sigmoid(zL)
    c["Z"+str(L)] = zL
    c["A"+str(L)] = aL
    return aL, c

def loss(y, yh):
    return -np.mean(y*np.log(yh+1e-8) + (1-y)*np.log(1-yh+1e-8))

def backward(x, y, p, c):
    g = {}
    m = x.shape[0]
    L = len(p)//2
    aL = c["A"+str(L)]
    dz = aL - y
    a_prev = c["A"+str(L-1)] if L > 1 else x
    g["dW"+str(L)] = (a_prev.T @ dz)/m
    g["db"+str(L)] = np.sum(dz, axis=0, keepdims=True)/m
    for i in reversed(range(1, L)):
        da = dz @ p["W"+str(i+1)].T
        dz = da * dtanh(c["Z"+str(i)])
        a_prev = x if i == 1 else c["A"+str(i-1)]
        g["dW"+str(i)] = (a_prev.T @ dz)/m
        g["db"+str(i)] = np.sum(dz, axis=0, keepdims=True)/m
    return g

def update(p, g, lr):
    L = len(p)//2
    for i in range(1, L+1):
        p["W"+str(i)] -= lr * g["dW"+str(i)]
        p["b"+str(i)] -= lr * g["db"+str(i)]
    return p

def acc(y, yh):
    yp = (yh >= 0.5).astype(int)
    return np.mean(yp == y)

def grad_norm(g):
    s = 0
    for k in g:
        s += np.sum(g[k]**2)
    return np.sqrt(s)

dims = [2, 32, 32, 16, 1]
params = init_params(dims)

epochs = 300
lr = 0.05

history = []

for e in range(epochs):
    yh, cache = forward(X_train, params)
    l = loss(y_train, yh)
    grads = backward(X_train, y_train, params, cache)
    params = update(params, grads, lr)

    yh_val, _ = forward(X_val, params)
    l_val = loss(y_val, yh_val)

    history.append([
        e,
        l,
        l_val,
        acc(y_train, yh),
        acc(y_val, yh_val),
        grad_norm(grads)
    ])

history_df = pd.DataFrame(history, columns=[
    "epoch",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "grad_norm"
])

yh_test, _ = forward(X_test, params)
test_accuracy = acc(y_test, yh_test)

print("Test Accuracy:", test_accuracy)
print(history_df.tail())