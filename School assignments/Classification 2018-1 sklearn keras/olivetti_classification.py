# -*- coding: utf-8 -*-


#Importing libriaries
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn import metrics
import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


#Importing Olivetti dataset and splitting it into train and test sets.
data = fetch_olivetti_faces(shuffle=True)
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


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
Plottin our dataset
Some parts of the following code were inspired by
http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
"""
#Loading visualisation tools
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

#Plotting our dataset
plt.scatter(X.T[0], X.T[1], c='b', **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)


"""
Plotting results of classification.
Training time and accuracy score, received after classifying our data using different
classification algorithms, are compared on the plot.
Some parts of the following code were inspired by
http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
"""
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(3)]

clf_names, score, training_time = results
training_time = np.array(training_time) / np.max(training_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')

plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()