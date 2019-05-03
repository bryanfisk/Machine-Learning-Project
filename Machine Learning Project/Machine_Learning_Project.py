import os
import random
from scipy import misc
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

def input_images(directory):
    X = []
    y = []
    count = 0
    for r, d, f in os.walk(directory):
        for file in f:
            two_percent = len(f) / 50
            blocks = int((count + 1) / two_percent)
            if int(count % two_percent) == 0:
                os.system('cls')
                print('Reading: ' + directory)
                print('█' * blocks, '-' * (50 - blocks), int((count + 1) / len(f) * 100), '%\t', sep = '')
            count += 1
            image = misc.imread(directory + '\\' + file, flatten = True)
            reduced_image = misc.imresize(image, (192, 192, 1))
            X.append(reduced_image)
            if 'virus' in file or 'bacteria' in file:
                y.append(1)
            else:
                y.append(0)
    return X, y

root = "D:\\Desktop\\chest_xray"
dirs = ['\\train\\NORMAL',
        '\\train\\PNEUMONIA',
        '\\test\\NORMAL',
        '\\test\\PNEUMONIA',
        '\\val\\NORMAL',
        '\\val\\PNEUMONIA']

X_train_n, y_train_n = input_images(root + dirs[0])
X_train_p, y_train_p = input_images(root + dirs[1])
X_test_n, y_test_n = input_images(root + dirs[2])
X_test_p, y_test_p = input_images(root + dirs[3])
X_val_n, y_val_n = input_images(root + dirs[4])
X_val_p, y_val_p = input_images(root + dirs[5])

X_train = np.asarray(X_train_n + X_train_p)
y_train = np.asarray(y_train_n + y_train_p)
X_test = np.asarray(X_test_n + X_test_p)
y_test = np.asarray(y_test_n + y_test_p)

batch_size = 40
num_classes = 2
epochs = 12
img_rows, img_cols = 192, 192

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#params = {'activation' : ['logistic', 'tanh', 'relu'],
#          'alpha' : [0.01, 0.001, 0.0001, 0.00001],
#          'learning_rate' : ['constant', 'adaptive'],
#          'learning_rate_init' : [0.01, 0.001, 0.0001]}
#grid = GridSearchCV(clf, params, cv = 5)
#grid.fit(X_train, y_train)
#print(grid.best_estimator_)
#grid.best_estimator_.fit(X_train, y_train)
#print(grid.best_estimator.score(X_train, y_train)) 

#[print(len(k)) for k in train_normal_X]

