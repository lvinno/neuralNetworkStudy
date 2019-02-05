#  卷积神经网络加多重感知器加反向传播

# import libraries and utils
import numpy as np
import pandas as pd
from keras.utils import np_utils
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from keras.datasets import mnist

# load data 
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# reshape 28x28 to 4D metrics 
x_train4D = x_train.reshape(60000,28,28,1).astype('float32')
x_test4D = x_test.reshape(10000,28,28,1).astype('float32')

# normalization
x_train4D_normalized = x_train4D/255
x_test4D_normalized = x_test4D/255

# one hot encoding 
y_trainOne_Hot = np_utils.to_categorical(y_train)
y_testOne_Hot = np_utils.to_categorical(y_test)

#import model and layer library from keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

# initialize model
model = Sequential()


'''
network:
conv   16 5x5 filter using relu

maxpooling  2x2

conv   36 5x5 filter using relu

maxpooling 2x2

flatten 7x7x36=1764   0 0 0 0 0 0 ...
    
middle  128             0 0 0

output  10            0  0 0 0 0 0

'''
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding="same",
                 input_shape=(28,28,1),
                 activation='relu'))


model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding="same",
                 input_shape=(28,28,1),
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation="relu"))

model.add(Dense(10,activation="softmax"))


# define training methods

model.compile(loss='categorical_crossentropy',
             optimizer='adam',metrics=['accuracy'])

# train the model
train_history = model.fit(x=x_train4D_normalized,
                          y=y_trainOne_Hot,
                          validation_split=0.2,
                          epochs=10,
                          batch_size=300,
                          verbose=2)

