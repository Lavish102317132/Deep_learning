import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
X = np.random.rand(1000, 3)
y = 3*X[:,0] + 2*X[:,1] - X[:,2] + np.random.randn(1000) * 0.1
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(3,)),  
    layers.Dense(32, activation='relu'),                   
    layers.Dense(1)                                        
])
model.compile(
    optimizer='adam',
    loss='mse',              
    metrics=['mae']          
)
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
loss, mae = model.evaluate(X, y)
print("MSE:", loss)
print("MAE:", mae)

new_sample = np.array([[0.5, 0.2, 0.1]])
prediction = model.predict(new_sample)
print("Prediction:", prediction)