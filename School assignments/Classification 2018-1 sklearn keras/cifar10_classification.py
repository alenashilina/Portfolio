# -*- coding: utf-8 -*-

#Importing libriaries
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import metrics

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Unpickling data and spliting it to train and test sets
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data1 = unpickle("cifar-10-batches-py/data_batch_1")
data2 = unpickle("cifar-10-batches-py/data_batch_2")
data3 = unpickle("cifar-10-batches-py/data_batch_3")
data4 = unpickle("cifar-10-batches-py/data_batch_4")
data5 = unpickle("cifar-10-batches-py/data_batch_5")
data_test = unpickle("cifar-10-batches-py/test_batch")

X_train = data1[b'data']
X_train = np.append(X_train, data2[b'data'], axis = 0)
X_train = np.append(X_train, data3[b'data'], axis = 0)
X_train = np.append(X_train, data3[b'data'], axis = 0)
X_train = np.append(X_train, data5[b'data'], axis = 0)

y_train = data1[b'labels']
y_train = np.append(y_train, data2[b'labels'], axis = 0)
y_train = np.append(y_train, data3[b'labels'], axis = 0)
y_train = np.append(y_train, data3[b'labels'], axis = 0)
y_train = np.append(y_train, data5[b'labels'], axis = 0)

X_test = data_test[b'data']
y_test = data_test[b'labels']

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#########################################################
"""
Classification part
Some parts of the following code were inspired by
http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
"""

"""
A function which accepts as a parameter the name of classification algorithm 
we are going to use and necessitated arguments.
Then we are fitting our classification algorithms to a dataset and count the accurasy of how well
our data were classified.
The function returns description of the current classification algorithm,
accuracy score and the value of time spent on training.
"""
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    print()
    
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time

"""
Going through different classification algorithms.
For each algorithm printing the name of it and then passing its name and parameters
to benchmark() function. Returned values we append to results[].
"""
results = []
for clf, name in (
        (Perceptron(n_iter=50), "Perceptron"),
        (KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p = 2), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest"),
        (GaussianNB(), "Gaussian Naive Bayes"),
        (SVC(kernel = 'linear', random_state = 0), "Support Vector Machine")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

##########################################################################
#Plotting part
    
"""
Plotting results of classification.
Training time and accuracy score, received after classifying our data using different
classification algorithms, are compared on the plot.
Some parts of the following code were inspired by
http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
"""

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()