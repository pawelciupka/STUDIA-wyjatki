import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

parser = argparse.ArgumentParser()
parser.add_argument('--p', required=True, type=str, help='Path to data file')
parser.add_argument('--cfg', required=True, type=str, help='Path to config file')
parser.add_argument('--a', required=True, type=str, help='Algorithm')

# Arguments for DBSCAN algorithm
parser.add_argument('--eps', default=0.5, type=float, help='DBSCAN - The maximum distance between two samples')
parser.add_argument('--min_s', default=10, type=float, help='DBSCAN - The number of samples in a neighborhood')

# Arguments for LOF algorithm
parser.add_argument('--n_neighbors', default=20, type=float, help='LOF - The number of neighbors')
parser.add_argument('--contamination', default=0.1, type=float, help='LOF - The value of contamination')

args = parser.parse_args()
data = pd.read_csv(args.p)

with open(args.cfg) as json_file:
    config = json.load(json_file)

X = data[config['x']]
X = StandardScaler().fit_transform(X)

if args.a == 'DBSCAN':
    db = DBSCAN(eps=args.eps, min_samples=args.min_s)
    y_pred = db.fit(X)
    labels = db.labels_
    
    plt.title("DBSCAN")


elif args.a == 'LOF':
    lof = LocalOutlierFactor(n_neighbors=args.n_neighbors, contamination=args.contamination)    
    y_pred = lof.fit_predict(X)
    labels = lof.negative_outlier_factor_

    plt.title("LOF")


if len(set(labels)) > 1:
    print(f"Quality: {davies_bouldin_score(X, labels)}")
    print("Silhouette Coefficient: %0.3f" % silhouette_score(X, labels))
    print("Calinski-Harabasz score: %0.3f" % calinski_harabasz_score(X, labels))
print(f"n_clusters_: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"n_noise_: {list(labels).count(-1)}")

plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')

radius = (labels.max() - labels) / (labels.max() - labels.min())
plt.scatter(X[:, 0], X[:, 1], s=500 * radius, edgecolors='r',
            facecolors='none', label='Outlier scores')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))

legend = plt.legend(loc='upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]

plt.show()