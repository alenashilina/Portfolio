# -*- coding: utf-8 -*-

#Importing libriaries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import mode
from sklearn import metrics
from sklearn.metrics import accuracy_score

#Loading visualisation tools
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}


"""
A function which accepts parameters such as our dataset, the name of clustering method
we are going to use and necessitated arguments.
Then we fitting our clustering methods to a dataset and count the accurasy of how well
our data was clustered, using function accuracy_count. Also we count V-measure. 
Then plot the result and display accuracy, V-measure value and computational time.
Some parts of the following code were inspired by
http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
"""
def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    y_pred = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    
    acs = accuracy_count(y_pred)
    v_meas = metrics.v_measure_score(y_pred, original_labels)
    palette = sns.color_palette('deep', np.unique(y_pred).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in y_pred]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    print('Clustering took {:.2f} s'.format(end_time - start_time))
    print('Accuracy score is ' + str(int(acs*100)) + '%')
    print('V-measure value: ' + str(v_meas))
    
    
"""
A function wich accepts as parameter predicted clusters. First we match each learned
cluster label with the true labels found in them. Then we count the accuracy of how
well our data was clastered.
This part of code was taken from the book "Python Data Science Handbook" by Jake VanderPlas
and then modified for needs of our task.
"""
def accuracy_count(y_res):
    new_labels = np.zeros_like(y_res)
    for i in range(6):
        mask = (y_res == i)
        new_labels[mask] = mode(original_labels[mask])[0]
    
    accuracy = accuracy_score(original_labels, new_labels)
    return accuracy
   
    
"""
Importing six categories of the 20NewsGroup dataset with the subset equaled 'test',
to reduce quantity of data to process. With that we obtain original lables for each entry.
"""
categories = [
        'comp.sys.mac.hardware',
        'rec.autos',
        'comp.graphics',
        'sci.space',
        'rec.sport.hockey',
        'sci.med',
    ]

text_20ng = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42, remove=())
original_labels = text_20ng.target


"""
Extracting features from the training data using a sparse vectorizer
Some parts of the following code were inspired by
http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
"""
def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6
data_size_mb = size_mb(text_20ng.data)
print("Extracting features from the training data using a sparse vectorizer")
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X = vectorizer.fit_transform(text_20ng.data)
print("n_samples: %d, n_features: %d" % X.shape)
X = X.toarray() 

#Plotting our dataset
plt.scatter(X.T[0], X.T[1], c='b', **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)


#Call our function for different methods of clustering and numbers of clusters and pass parameters to it

#Mini Batch KMeans clustering algorithm
plot_clusters(X, cluster.MiniBatchKMeans, (), {'n_clusters':6, 'init':'k-means++', 'n_init':10, 'random_state':0})

#Birch clustering algorithm
plot_clusters(X, cluster.Birch, (), {'n_clusters':6, 'threshold':0.5, 'branching_factor':50})

#Agglomerative Clustering algorithm
plot_clusters(X, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'affinity': 'euclidean', 'linkage':'ward'})

#KMeans clustering algorithm
plot_clusters(X, cluster.KMeans, (), {'n_clusters':6, 'init':'k-means++', 'max_iter':300, 'n_init':10, 'random_state':0})

#Spectral Clustering clustering algorithm
plot_clusters(X, cluster.SpectralClustering, (), {'n_clusters':2, 'random_state':0, 'n_init':10, 'affinity':'rbf'})
