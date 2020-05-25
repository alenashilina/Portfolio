# -*- coding: utf-8 -*-

#Importing libraries
import tensorflow as tf
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Defining parameters
n_classes = 40
batch_size = 32
epochs = 100

#Importing Olivetti dataset
data = fetch_olivetti_faces(shuffle=True)

data_x = data.data
data_y = data.target


#Splitting the dataset to the train and test set
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.25, random_state = 0)

#Turning our lables to categorical variables
y_train = tf.one_hot(y_train, n_classes, dtype = 'int32')
y_test = tf.one_hot(y_test, n_classes, dtype = 'int32')

#Defining model architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
model.add(tf.keras.layers.Dense(4096, activation = 'relu'))
model.add(tf.keras.layers.Dense(n_classes, activation = "softmax"))

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

#Fitting DNN to our data and getting results
history = model.fit(x_train,y_train, batch_size = batch_size,
                    epochs = epochs, verbose = 1, validation_data = (x_test, y_test))
score = model.evaluate(x_test, y_test, batch_size = batch_size, verbose = 1)

print('Test score:', score[0])
print('Test accuracy:', score[1])

#Listing data in history
print(history.history.keys())

#Summarizing history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Summarizing history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()