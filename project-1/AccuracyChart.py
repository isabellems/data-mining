import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from random import *
from numpy.linalg import *
from time import *
import matplotlib.pyplot as plt

numComponents = [20,40,60,80,100]
li = []

for nC in numComponents:
	time1 = time()
	inputF = pd.read_csv('train_set.csv',sep='\t')
	vectorizer = TfidfVectorizer(stop_words='english')
	data = inputF['Title'] + inputF['Content']
	vectorizer.fit_transform(data)
	X_train = vectorizer.transform(data)
	svd = TruncatedSVD(n_components=nC, random_state = 42)
	X_lsi = svd.fit_transform(X_train)
	categories = inputF['Category']
	labelEnc = preprocessing.LabelEncoder()
	labelEnc.fit(categories)
	Y_train = labelEnc.transform(categories)

	randFor_ = RandomForestClassifier(n_estimators = 17, criterion = 'entropy')
	totalAcc = 0
	num_of_splits = 10
	kf = KFold(len(X_lsi),n_folds=num_of_splits)

	for train_index, test_index in kf:
		X_train_counts = X_lsi[train_index]
		X_test_counts = X_lsi[test_index]
		Y_train_counts = Y_train[train_index]
		Y_test_counts = Y_train[test_index]
		randFor_.fit(X_train_counts, Y_train_counts)
		Y_pred = randFor_.predict(X_test_counts)
		acc = accuracy_score(Y_test_counts,Y_pred)
		totalAcc += acc

	li.append(totalAcc/num_of_splits)
	time2 = time()
	diff = time2 - time1
	print ("Number of components %d took %0.2f seconds" % (nC,diff))


print (li)
plt.plot(numComponents,li)
plt.axis([10,110,0.9,1])
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.savefig('ComponentAccuracy.png')
plt.clf()
#plt.show()

