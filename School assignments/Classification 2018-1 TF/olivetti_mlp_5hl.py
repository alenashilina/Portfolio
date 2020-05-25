# -*- coding: utf-8 -*-

#Importing libraries
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cross_validation import train_test_split

#Importing Olivetti dataset
data = fetch_olivetti_faces(shuffle=True)
data_x = data.images
data_y = data.target

#The function to reshape our inputs
def reshape_x (input_data):
    new_x = np.array([])
    for i in range (len(input_data)):
        new_arr = np.array([])
        for j in range (64):
            for n in range (64):
                res = 0
                res = data_x[i][j][n]
                new_arr = np.append(new_arr, [res])
        new_x = np.append(new_x, [new_arr])
        
    reshaped_data = np.reshape(new_x, (400, 4096))
    return reshaped_data

#Splitting the dataset to the train and test set
x_reshaped = reshape_x(data_x)
x_train, x_test, y_train, y_test = train_test_split(x_reshaped, data_y, test_size = 0.25, random_state = 0)

#Defining Network Parameters
n_hidden_1 = 4096 
n_hidden_2 = 4096 
n_hidden_3 = 4096
n_hidden_4 = 4096 
n_hidden_5 = 4096 
n_classes = 40
num_epochs = 100

#Turning our lables to categorical variables
y_train = tf.one_hot(y_train, n_classes, dtype = 'int32')
y_test = tf.one_hot(y_test, n_classes, dtype = 'int32')


#Defining placeholders
x = tf.placeholder("float", [None, 4096])
y = tf.placeholder("float", [None, n_classes])


#The funcion to model the network
def neural_network_model (data):
    
    #Initialising weights and biases for each layer
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([4096, n_hidden_1])),
                      'biases': tf.Variable(tf.random_normal([n_hidden_1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                      'biases': tf.Variable(tf.random_normal([n_hidden_2]))}
    
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
                      'biases': tf.Variable(tf.random_normal([n_hidden_3]))}
    
    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
                      'biases': tf.Variable(tf.random_normal([n_hidden_4]))}
    
    hidden_5_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
                      'biases': tf.Variable(tf.random_normal([n_hidden_5]))}
    
    output_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_5, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    #Defining layers of the network
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)
    
    l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
    l5 = tf.nn.relu(l5)
    
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    
    return output


#The function to train the network
def train_neural_network(x):
    #Getting predictions for the data
    prediction = neural_network_model(x)
    #Calculating the cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    #Applying Adam Optimizer
    optimizer  = tf.train.AdamOptimizer().minimize(cost)
    
    n_epochs = num_epochs
    #Running the tensorflow session
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        #training the data
        for epoch in range(n_epochs):
            epoch_loss = 0
            epoch_x = x_train
            epoch_y = sess.run(y_train)
            _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y}) #c is for cost
            epoch_loss += c
            print('Epoch ', epoch, 'complited out of ', n_epochs, '. Loss: ', epoch_loss)
        
        #Evaluating the model
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        test_label = sess.run(y_test)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float')) 
        print ('Accuracy: ', accuracy.eval({x: x_test, y: test_label}))
    
#Running the network  
train_neural_network(x)