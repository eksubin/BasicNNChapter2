import sys
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout

##
#loading the text
I = requests.get('https://cs.stanford.edu/people/karpathy/char-rnn/warpeace_input.txt')
raw_text = I.text

##
chars = sorted(list(raw_text))
print("number of chrs : ", len(chars))

##
ix_to_char = {ix: char for ix, char in enumerate(chars)}
char_to_ix = {char: ix for ix, char in enumerate(chars)}

maxlen = 10
x_data = []
y_data = []
for i in range(0, len(raw_text)- maxlen, 1):
    in_seq = raw_text[i : i+maxlen]
    out_seq = raw_text[i + maxlen]
    x_data.append(char_to_ix[char] for char in in_seq)
    y_data.append(char_to_ix[out_seq])
n_chars = len(x_data)
print(x_data[1])

##
print("number of chara : ", n_chars)
#datascaling and reshaping
n_vocab = len(chars)
x = np.reshape(x_data, (n_chars, maxlen, 1))
x = x / float(n_vocab)

y = tf.keras.utils.to_categorical(y_data)
##
model = Sequential()
model.add(LSTM(800, input_shape=(len(x[1]), 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(800, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(800))
model.add(Dropout(0.2))
model.add(Dense(len(y[1]), activation='softmax'))
model.summary()

##

