from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D
from keras.optimizers import SGD
from keras.datasets import imdb
from keras import backend as K


# set parameters:
max_features = 5000
maxlen = 100
batch_size = 32
embedding_dims = 50
conv_dim = 250
window = 3
hidden_dims = 250
nb_epoch = 2

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                      test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))

# we add a Convolution1D, which will learn conv_dim
# word group filters of size window:
model.add(Convolution1D(nb_filter=conv_dim,
                        filter_length=window,
                        border_mode='valid',
                        activation='tanh',
                        subsample_length=1))

# we use max over time pooling by defining a python function to use
# in a Lambda layer
def max_1d(X):
    return K.max(X, axis=1)

model.add(Lambda(max_1d, output_shape=(conv_dim,)))

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))

model.add(Activation('tanh'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
