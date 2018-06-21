# -*- coding: utf-8 -*-
"""
Created on Fri May 25 03:56:48 2018

@author: farhan
"""

from __future__ import print_function
import numpy as np

from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
import matplotlib as plt
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


#Loading Cifar10 data
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

num_classes = 10

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(1024, activation='relu')) #to add non linearity
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#compile model
model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, decay=1e-6),
                  metrics=['accuracy'])


#Adding check points
filepath="'D:\newsavedModel.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#training
Model = model.fit(X_train / 255.0, to_categorical(Y_train),batch_size=128,
              shuffle=True,
              epochs=30,
              validation_data=(X_test / 255.0, to_categorical(Y_test)),
              callbacks=[EarlyStopping(min_delta=0.001, patience=3)])

# Evaluate the model
scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))


#print Accuracy and loss
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])



#Accuracy curve
plt.plot(Model.history['acc'])
plt.plot(Model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Loss curve
plt.plot(Model.history['loss'])
plt.plot(Model.history['val_loss'])
plt.legend(['Training loss', 'TestLoss'])
plt.xlabel('Epochs ')
plt.ylabel('Loss')
plt.title('Loss Curves')

#Model Summary
model.summary()

model.save('D:\FarhanfinalModel.h')

new_model = load_model('D:\FarhanfinalModel.h')


 #Prediction Function
class_pred =model.predict(X_test)
print(class_pred[1])

#  0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',  4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck',
