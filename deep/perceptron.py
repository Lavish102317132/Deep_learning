import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Perceptron
or_data = pd.DataFrame({
    'input1': [0, 0, 1, 1],
    'input2': [0, 1, 0, 1],
    'output': [0, 1, 1, 1]
})

X_or = or_data[['input1', 'input2']]
y_or = or_data['output']
clf_or = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
clf_or.fit(X_or, y_or)
sns.scatterplot(
    data=or_data,
    x='input1',
    y='input2',
    hue='output',
    s=200
)


w1, w2 = clf_or.coef_[0]
b = clf_or.intercept_[0]

x = np.linspace(-0.2, 1.2, 100)
y = -(w1 * x + b) / w2

plt.plot(x, y, color='black')
plt.title("OR Gate - Learned Decision Boundary")
plt.show()
and_data = pd.DataFrame({
    'input1': [0, 0, 1, 1],
    'input2': [0, 1, 0, 1],
    'output': [0, 0, 0, 1]
})

X_and = and_data[['input1', 'input2']]
y_and = and_data['output']
clf_and = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
clf_and.fit(X_and, y_and)
sns.scatterplot(
    data=and_data,
    x='input1',
    y='input2',
    hue='output',
    s=200
)


w1, w2 = clf_and.coef_[0]
b = clf_and.intercept_[0]

x = np.linspace(-0.2, 1.2, 100)
y = -(w1 * x + b) / w2

plt.plot(x, y, color='black')
plt.title("AND Gate - Learned Decision Boundary")
plt.show()