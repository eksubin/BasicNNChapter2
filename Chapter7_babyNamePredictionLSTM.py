import sys
import re
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint

##
I = requests.get('https://cs.stanford.edu/people/karpathy/namesGenUnique.txt')

##
raw_text = I.text
len(raw_text)

##
#preprocessing the text
raw_text = raw_text.replace('\n', ' ')
#extracting all the unique charectors
set(raw_text)

##
#remove all the unwanted charectors
raw_text = re.sub('[-.0-9:]', '', raw_text)
raw_text1 = raw_text.lower()
set(raw_text1)


##
#all the other unwanted charectors are removed
len1 = len(set(raw_text1))

##
#giving numbers to the charectors

chars = sorted(set(raw_text1))
arr = np.arange(0, len1)

char_to_ix = {}
ix_to_char = {}
for i in range(len1):
    char_to_ix[chars[i]] = arr[i]
    ix_to_char[arr[i]] = chars[i]

##
#Now we will create output sequence using the gnerated map

maxlen = 5
x_data = []
y_data = []
for i in range(0, len(raw_text1) - maxlen, 1):
    in_seq = raw_text1[i: i+maxlen]
    out_seq = raw_text1[i + maxlen]

    x_data.append([char_to_ix[char] for char in in_seq])
    y_data.append([char_to_ix[char] for char in out_seq])

print('Text corpus : ', len(x_data))
print('Sequences : ', len(x_data)/maxlen)

##
#Data reshaping and converting to catogorical
nb_chars = len(x_data)
x = np.reshape(x_data, (nb_chars, maxlen, 1))
x = x/float(len(chars))

y = tf.keras.utils.to_categorical(y_data)

##
model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, 1), return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(len(y[1]), activation='softmax'))
print(model.summary())

##
model.compile(loss='categorical_crossentropy', optimizer='adam')
#creating model checkpoint
filepath = "model_weights_babynames.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True,mode=min)
model_callbacks = [checkpoint]


##
model.fit(x, y, epochs=5, batch_size=32, callbacks=model_callbacks)

##
#prediction

pattern = []
seed = 'handi'
for i in seed:
    value = char_to_ix[i]
    pattern.append(value)
n_vocab = len(chars)


##
output = []
for i in range(100):
    X = np.reshape(pattern, (1,len(pattern), 1))
    X = X / float(n_vocab)

    int_prediction = model.predict(X, verbose=0)
    index = np.argmax(int_prediction)
    prediction = ix_to_char[index]
    #sys.stdout.write(prediction)
    output.append(prediction)
    #pattern = pattern[1:len(pattern)]

##

