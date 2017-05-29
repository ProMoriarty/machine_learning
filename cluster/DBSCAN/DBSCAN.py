from sklearn.cluster import DBSCAN
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

mac2id = {}
onlinetimes = []
fopen = open("学生月上网时间分布-TestData.txt",encoding="utf-8")
for line in fopen:
	items = line.split(",")
	mac = items[2]
	onlinetime = items[6]
	starttime = int(items[4].split(" ")[1].split(":")[0])
	if mac not in mac2id:
		mac2id[mac] = len(onlinetime)
		onlinetimes.append((starttime,onlinetime))
	else:
		onlinetimes[mac2id[mac]] = [(starttime,onlinetime)]

real_X = np.array(onlinetimes).reshape((-1,2))
X = real_X[:,0:1]

db = DBSCAN(eps=0.01, min_samples=20).fit(X)
labels = db.labels_

raito = len(labels[labels[:]==-1])/len(labels)
print("Noise raitio:",format(raito,".2%"))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print("Estimated number of clusters: %d" %n_clusters_)
print("Silhouette Coefficient:%0.3f" %metrics.silhouette_score(X, labels))

for i in range(n_clusters_):
	print("Cluster",i,":")
	print(list(X[labels==i].flatten()))

