"""
k-means clustering demo..taken from
https://github.com/scikit-learn/scikit-learn/blob/master/examples/cluster/plot_cluster_iris.py
"""
#prints the above doc 
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
#not used but required for 3d projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

print("starting program")
np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target

estimators = [('k_means_iris_8', KMeans(n_cluster=8)),
              ('k_means_iris_3', KMeans(n_cluster=3)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1, init='random') )]

fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']

for name, est in estimators:
    fig = plt.figure(fignum, figsize(4,3))
    ax = Axes3D(fig, rect=[0,0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_