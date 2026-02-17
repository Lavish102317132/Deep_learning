import pandas as pd
import numpy as np
pip install ucimlrepofrom ucimlrepo 
import fetch_ucirepo 
  
# fetch dataset 
df = fetch_ucirepo(id=1) 
  
# data (as pandas dataframes) 
X = df.data.features 
y = df.data.targets
X.head()
y.head()
y=y+1.5
y.head()
#features are shell_weight, length,height
X=X.drop(columns=['Sex','Diameter','Whole_weight','Shucked_weight','Viscera_weight'])
X.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.2,
                                                random_state=42)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
std[std == 0] = 1

# Normalize
x_train= (x_train-mean)/std
x_test= (x_test-mean)/std
def forward(X,w,b):
    """
    computes y_hat = Xw + b
    """
    y_hat =X @ w+ b   # matrix multiplication

    # shapes
    print("Shape of X:", X.shape)
    print("Shape of w:", w.shape)
    print("Shape of b:", np.shape(b))
    print("Shape of y_hat:", y_hat.shape)

    return y_hat
def mse(y, y_hat):
    #MSE
    loss = np.mean((y -y_hat)**2)
    return loss
def grad_w(X,y,y_hat):
    """
    returns gradient with respect to w
    shape must match w
    """
    n=X.shape[0]
    dW=(2/n)*(X.T @ (y_hat-y))
    return dW


def grad_b(y,y_hat):
    """
    returns gradient with respect to b
    """
    n=y.shape[0]
    db=(2/n)*np.sum(y_hat-y)
    return db
n_features = x_train.shape[1]

w= np.random.randn(n_features)*0.01   # small random values
b= 0                                  
# Hyperparameters
learning_rate = 0.01
epochs = 1000
k=100

for epoch in range(epochs):

    
    y_hat =forward(x_train,w,b)

    
    loss =mse(y_train,y_hat)

    
    dW =grad_w(x_train,y_train,y_hat)
    db =grad_b(y_train,y_hat)

    
    w =w- learning_rate*dW
    b =b- learning_rate*db

    if epoch % k == 0: