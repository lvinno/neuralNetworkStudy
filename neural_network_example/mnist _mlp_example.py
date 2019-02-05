# 多层感知器加反向传播 example

# import libraries and utils
import numpy as np
import pandas as pd
from keras.utils import np_utils
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


np.random.seed(10)

# load data from mnist which is a dataset in keras 
from keras.datasets import mnist
(x_train_image,y_train_label),(x_test_image,y_test_label) = mnist.load_data()

# check the data length and training data shape
print('train_data= ',len(x_train_image))
print('train_label= ',len(y_train_label))
print('train_image=',x_train_image.shape)
print('test data =',len(x_test_image))



# show function for showing train image 
def plot_image(image):
    plt.imshow(image, cmap='binary')
    plt.show()
    

# plot_image(x_train_image[0])
print(x_train_image[0])

# reshape the 28x28 image into one dimension vector for total 60000 train data and 10000 test data

X_Train = x_train_image.reshape(60000,784).astype('float32')
X_Test = x_test_image.reshape(10000,784).astype('float32')

print(X_Train[0])

# normalize the data based on rgb standard 255 for each point
x_Train_normalize = X_Train/ 255
x_Test_normalize = X_Test/ 255


# look at the label data
print(y_train_label[:5])

# use np_utils.to_categorial to transfer the data 
y_TrainOne_Hot = np_utils.to_categorical(y_train_label)
y_TestOne_Hot = np_utils.to_categorical(y_test_label)

print(y_TrainOne_Hot[:5])

# import Sequential and Dense for defining the model and layer
from keras.models import Sequential
from keras.layers import Dense  
from keras.layers import Dropout

# now we create a model, we can add different dense inside it
model = Sequential()

# define the input number as 784, middle hidden layer as 256
'''
input  0  0  0  0  784

middle1   0  0   1000

output 0 0  0 0 0 0 10

'''
model.add(Dense(units=1000,
                input_dim=784,
                kernel_initializer='normal',
                activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))
print(model.summary())


# now we can train our model using complie 
# to define loss function , optimizer and metrics
#
# and fix to 

model.compile(loss='categorical_crossentropy',
                optimizer='adam',metrics=['accuracy'])


# put in x as feature, y as label,
#  validation_split means seperate 0.2 data for validation
# epoch means 10 rounds
# batch_size means 200 times per round
# verbose=2 means show the training process

train_history = model.fit(x=x_Train_normalize,
                          y=y_TrainOne_Hot,
                          validation_split=0.2,
                          epochs=10,
                          batch_size=200,
                          verbose=2)

# after training use model.evaluate to use the test data and label see the accuravy 
scores = model.evaluate(x_Test_normalize,y_TestOne_Hot)
print('accuracy = ',scores)


