
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix
from random import *
from numpy.linalg import *
from time import *


def initCenters(vector,K):
	shape = vector.shape[0]
	random = sample(range(shape),K)
	means = [vector[i] for i in random]
	return means

def newCenters(vector,clusters):
	means = []
	for cl in clusters:
		length = len(cl)
		if length != 0:
			x = [vector[item] for item in cl]
			mean = sum(x)/length
		else:
			x = randrange(vector.shape[0])
			mean = vector[x]
		means.append(mean)
	return means

def kMeans(vector,K):
	means = initCenters(X_lsi,K)
	end = False
	prevClusters = []
	while end == False :
		clusters = [[] for cl in range(K)]
		for i,item in enumerate(vector):
			for m,clItem in enumerate(means):
				dist = cosine(item,clItem)
				if m == 0:
					minD = dist
					minI = 0
				else:
					if dist < minD:
						minD = dist
						minI = m
			clusters[minI].append(i)
		newMeans = newCenters(vector,clusters)
		if clusters == prevClusters:
			end = True 
		else:
			prevClusters = clusters
			means = newMeans
	return (clusters)

def percentages(clusters, categories):
	numbers = {'Politics':0, 'Film':0, 'Football':0, 'Business':0, 'Technology':0}
	n = len(clusters)
	columns = ['Politics', 'Film', 'Football', 'Business', 'Technology']
	rows = []
	for i in range(n):
	 	string = "Cluster %d" %(i+1)
	 	rows.append(string)
	df_ = pd.DataFrame(index=rows,columns=columns)
	df_ = df_.fillna(0.0)
	for i,cl in enumerate(clusters):
		length = len(cl)
		for item in cl:
			x = categories[item]
			df_[x][i] += 1.0
		for cat in columns:
			x = df_[cat][i] / length
			df_[cat][i] = '%.3f' % x
	print (df_)
	output = 'clustering_KMeans.csv'
	df_.to_csv(output, sep='\t', encoding='utf-8')

time1 = time()
inputF = pd.read_csv('train_set.csv',sep='\t')
vectorizer = TfidfVectorizer(stop_words='english')
data = inputF['Title'] + inputF['Content']
vectorizer.fit_transform(data)
X_train_tfidf = vectorizer.transform(data)
svd = TruncatedSVD(n_components=90, random_state=21)
X_lsi = svd.fit_transform(X_train_tfidf)
categories = inputF['Category']

km = kMeans(X_lsi,5)
percentages(km, categories)
time2 = time()
diff = time2 - time1
print ('Ended in ' + str(diff) +  ' seconds')