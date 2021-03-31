##
import numpy as np
import tensorflow as tf

##
from tensorflow.python.keras.callbacks import History

number_of_datapoint = 100
# generate the datapoints
x = np.random.uniform(low=-5, high=-5, size=(number_of_datapoint, 1))
y = np.random.uniform(low=-5, high=-5, size=(number_of_datapoint, 1))
noise = np.random.uniform(low=-1, high=1, size=(number_of_datapoint, 1))

z = 7 * x + 6 * y + 5 + noise

##
input_z = np.column_stack((x, y))
print(input_z.shape)

##
# Network structure
model = tf.keras.Sequential(tf.keras.layers.Dense(units=1, input_shape=[100, 2]))
##
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mse'])

##
history = History()
model.fit(input_z, z, epochs=5, verbose=1, validation_split=0.2, callbacks=[history])
##
# visualization
print(history.history.keys())

##
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss graph')
plt.show()

##

