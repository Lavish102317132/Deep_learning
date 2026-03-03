import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

N = 1000
X1 = np.random.randn(N//2, 2) + np.array([2, 2])
X2 = np.random.randn(N//2, 2) + np.array([-2, -2])

X = np.vstack([X1, X2])
y = np.vstack([np.ones((N//2,1)), np.zeros((N//2,1))])

perm = np.random.permutation(N)
X = X[perm]
y = y[perm]

train_end = int(0.7 * N)
val_end = int(0.85 * N)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

def init_params(dims):
    p = {}
    for i in range(1, len(dims)):
        p["W"+str(i)] = np.random.randn(dims[i-1], dims[i]) * np.sqrt(2/dims[i-1])
        p["b"+str(i)] = np.zeros((1, dims[i]))
    return p

def relu(z):
    return np.maximum(0, z)

def drelu(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward(x, p):
    c = {}
    a = x
    L = len(p)//2
    for i in range(1, L):
        z = a @ p["W"+str(i)] + p["b"+str(i)]
        a = relu(z)
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
        dz = da * drelu(c["Z"+str(i)])
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

dims = [2, 12, 6, 1]
params = init_params(dims)

epochs = 200
lr = 0.1

tr_loss = []
vl_loss = []
tr_acc = []
vl_acc = []
gn = []

for e in range(epochs):
    yh, cache = forward(X_train, params)
    l = loss(y_train, yh)
    grads = backward(X_train, y_train, params, cache)
    params = update(params, grads, lr)

    yh_val, _ = forward(X_val, params)
    l_val = loss(y_val, yh_val)

    tr_loss.append(l)
    vl_loss.append(l_val)
    tr_acc.append(acc(y_train, yh))
    vl_acc.append(acc(y_val, yh_val))
    gn.append(grad_norm(grads))

yh_test, _ = forward(X_test, params)
print(acc(y_test, yh_test))

plt.figure()
plt.plot(tr_loss)
plt.show()

plt.figure()
plt.plot(vl_loss)
plt.show()

plt.figure()
plt.plot(tr_acc)
plt.show()

plt.figure()
plt.plot(vl_acc)
plt.show()

plt.figure()
plt.plot(gn)
plt.show()