import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

X = np.random.rand(100, 10, 1)
y = np.sum(X, axis=1)

model = Sequential()
model.add(GRU(32, input_shape=(10, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=10, batch_size=8)

test_input = np.random.rand(1, 10, 1)
print(model.predict(test_input))