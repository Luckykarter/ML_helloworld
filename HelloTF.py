import tensorflow as tf
import numpy as np
from tensorflow import keras
import datagen
from pprint import pprint as pp

# simple neural network - has 1 layer and 1 neuron
# the input shape has only one value
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# compile the model
# optimizer - Stochastic Gradient Descent
# loss - mean squared error

model.compile(optimizer='sgd', loss='mean_squared_error')

# providing the data
x_data = [float(x) for x in range(-1, 15)]
y_data = [datagen.hw_func(x) for x in x_data]

xs = np.array(x_data, dtype=float)
ys = np.array(y_data, dtype=float)

# training the neural network
model.fit(xs, ys, epochs=1000)

# make prediction on new dataset items
print(model.predict([20.0, 21.0, 22.0])[0])
print([datagen.hw_func(20.0), datagen.hw_func(21.0), datagen.hw_func(22.0)])

