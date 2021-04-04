import tensorflow as tf
import tensorflow_datasets as tdfs

##
print('number of dataset', tdfs.list_builders())
tdfs.list_builders()
##
# create traning and testing dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train.shape

##
import matplotlib.pyplot as plt

plt.imshow(x_train[42])

##
# creating validation set
from sklearn.model_selection import train_test_split

X_train, x_val, Y_train, y_val = train_test_split(x_train, y_train, random_state=0, test_size=0.5)
##
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=15,
                             width_shift_range=0.1,
                             height_shift_range=0.2,
                             horizontal_flip=True)


##
# datanormalization

def normalize(data):
    data = data.astype(tf.float32)
    data = data / 255
    return data


X_train = normalize(X_train)
datagen.fit(X_train)
x_val = normalize(x_val)
datagen.fit(x_val)

Y_train = tf.keras.utils.to_categorical(Y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
y_val = tf.keras.utils.to_categorical(y_val,10)
##
#for comparing multiple models use a function
from tensorflow.keras import models, layers
##

