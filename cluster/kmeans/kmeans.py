import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
def loadData(path):
	fopen = open(path,"r+")
	lines = fopen.readlines()
	retData = []
	retCityName = []
	for line in lines:
		items = line.strip().split(",")
		retCityName.append(items[0])
		retData.append([float(item) for item in items[1:]])
	return retData,retCityName

if __name__ == '__main__':
	data,cityName = loadData('city.txt')
	for i in range(3,10,1):
		km = KMeans(n_clusters = i)
		s = km.fit(data)
		center = km.cluster_centers_
		print (center)
		labels = km.labels_
		print(i,km.inertia_)
		filename = r"e:\1"
		fopen = open(filename,"w+")
		fopen.close()
		joblib.dump(km,filename)
		
		
		print(s)