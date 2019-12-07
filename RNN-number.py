import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.layers import SimpleRNN

# image resolution
img_rows, img_cols = 28, 28

TIME_STEPS = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001

# input data, in "train" and "test"
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# transform the original data
X_train = X_train.reshape(-1,img_rows,img_cols)/255
X_test = X_test.reshape(-1,img_rows,img_cols)/255

# transform the original data
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# dimension of input data
input_shape = (img_rows, img_cols, 1)


# build RNN model
model = Sequential()

#RNN cell
model.add(SimpleRNN(
            batch_input_shape=(None, img_rows, img_cols),
    units= 50,
    unroll=True,
       
        ))
# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

#optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#training
model.fit(X_train, y_train,
           batch_size=128 * 2,
            epochs=2,
             verbose=1,
              validation_data=(X_test, y_test))

#check
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
