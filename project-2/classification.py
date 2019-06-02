import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import operator
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.cross_validation import KFold
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix
from scipy import interp
from math import log
from sklearn import svm
from random import *
from numpy.linalg import *
from time import *

inputF = pd.read_csv('train.tsv',sep='\t')
new_input = inputF.copy()
categorical = []

for y,col in enumerate(inputF):
    if col=='Label':
        break
    if not str(inputF[col][0]).isnumeric():
        categorical.append(col)
X_train = pd.get_dummies(inputF, columns=categorical)
Y_train = X_train['Label']
del X_train['Label']
del X_train['Id']

good = 0
bad = 0

for label in Y_train:
    if label==1:
        good = good + 1
    else:
        bad = bad + 1

total = good + bad

good_perc = float(good)/total
bad_perc = float(bad)/total

testF = pd.read_csv('test.tsv',sep='\t')
X_test = pd.get_dummies(testF, columns=categorical)
ids = X_test['Id']
del X_test['Id']

del categorical[:]

svm_ = LinearSVC()
randFor_ = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
naiveB_ = BernoulliNB()

norm_X_train = (X_train - X_train.mean()) / (X_train.max() - X_train.min())

num_of_splits = 10
kf = KFold(len(norm_X_train),n_folds=num_of_splits)
cls_ = {'Naive Bayes': naiveB_, 'Random Forest':randFor_, 'SVM':svm_}
cols = ['Random Forest','Naive Bayes','SVM']
rows = ['Accuracy']
acc_df = pd.DataFrame(index=rows,columns=cols)

accList = []
best_acc = 0
best_cl = ' '

for classifier in cls_:
    print ("Classifier %s starting" %classifier)
    totalAcc = 0
    spl = 1
    for train_index, test_index in kf:
        print ("Fold %d" % spl)
        X_train_counts = norm_X_train.iloc[train_index]
        X_test_counts = norm_X_train.iloc[test_index]
        Y_train_counts = Y_train[train_index]
        Y_test_counts = Y_train[test_index]
        cls_[classifier].fit(X_train_counts, Y_train_counts)
        Y_pred = cls_[classifier].predict(X_test_counts)
        acc = accuracy_score(Y_test_counts,Y_pred)
        totalAcc += acc
        spl = spl + 1
    if totalAcc > best_acc:
        best_acc = totalAcc
        best_cl = classifier
    acc_df[classifier][0] = '%.3f' % (totalAcc/num_of_splits)

print ("The best classifier is " + best_cl)
accList.append(float(best_acc/num_of_splits))
output = 'EvaluationMetric_10fold.csv'
acc_df.to_csv(output, sep='\t', encoding='utf-8')

pred_dict = {'ID':[],'Predicted Category':[]}

cls_[best_cl].fit(norm_X_train,Y_train)
predictions = cls_[best_cl].predict(X_test)

for i,x in enumerate(predictions):
    pred_dict['ID'].append(ids[i])
    if x == 1:
        pred_dict['Predicted Category'].append('Good')
    else:
        pred_dict['Predicted Category'].append('Bad')

test_df = pd.DataFrame(data = pred_dict)
test_df = test_df.ix[::, ['ID', 'Predicted Category']]
test_df.to_csv('testSet_categories.csv', sep = '\t', index = False) 

totalEntropy = 0-good_perc*log(good_perc,2)-bad_perc*log(bad_perc,2)

infoGainList = []
attrDict = {}
labelDict = {}
for i,attr in enumerate(inputF):
    if attr=='Label':
        break
    if str(inputF[attr][0]).isnumeric():
        inputF[attr] = pd.cut(np.array(inputF[attr]), bins=5)
    set_ = set(inputF[attr])
    for s in set_:
        if s not in attrDict:
            attrDict[s] = 0
            labelDict[s] = 0
    for y,a in enumerate(inputF[attr]):
        attrDict[a] = attrDict[a] + 1
        if Y_train[y] == 1:
            labelDict[a] = labelDict[a] + 1
    attrEntropy = 0
    for a in attrDict:
        t1 = float((labelDict[a])/attrDict[a])
        t2 = float((attrDict[a]-labelDict[a])/attrDict[a])
        temp = float(attrDict[a])/total*(((0-t1)*log(t1,2))+((0-t2)*log(t2,2)))      
        attrEntropy = attrEntropy + temp
    infoGain = totalEntropy - attrEntropy
    infoGainList.append((attr,infoGain))
    attrDict.clear()
    labelDict.clear()
infoGainList = sorted(infoGainList, key=lambda x: x[1])

infoGDict = {'Attribute':[],'Information Gain':[]}
for a in infoGainList:
    infoGDict['Attribute'].append(a[0])
    infoGDict['Information Gain'].append(a[1])

infoF_df = pd.DataFrame(data = infoGDict)
infoF_df = infoF_df.ix[::, ['Attribute', 'Information Gain']]
infoF_df.to_csv('infoGain.csv', sep = '\t', index = False) 

remList = [0]
length = len(infoGDict['Attribute'])
for i,attr in enumerate(infoGDict['Attribute']):
    if i == length-1:
        break
    remList.append(i + 1)
    print ("Removing " + attr)
    del new_input[attr]
    del categorical[:]
    for y,col in enumerate(new_input):
        if col=='Label':
            break
        if not str(new_input[col][0]).isnumeric():
            categorical.append(col)
    new_X_train = pd.get_dummies(new_input, columns=categorical)
    new_Y_train = new_X_train['Label']
    del new_X_train['Label']
    del new_X_train['Id']


    new_norm_X_train = (new_X_train - new_X_train.mean()) / (new_X_train.max() - new_X_train.min())

    new_kf = KFold(len(norm_X_train),n_folds=num_of_splits)
    totalAcc = 0 
    for train_index, test_index in new_kf:
        X_train_counts = new_norm_X_train.iloc[train_index]
        X_test_counts = new_norm_X_train.iloc[test_index]
        Y_train_counts = new_Y_train[train_index]
        Y_test_counts = new_Y_train[test_index]
        cls_[best_cl].fit(X_train_counts, Y_train_counts)
        Y_pred = cls_[best_cl].predict(X_test_counts)
        acc = accuracy_score(Y_test_counts,Y_pred)
        totalAcc += acc
    tempAcc = float(totalAcc/num_of_splits)
    accList.append(tempAcc)
print ('Creating plot')

min_x = min(remList)
min_y = min(accList) - 0.02
max_x = max(remList)
max_y = max(accList) + 0.02

plt.plot(remList,accList)
plt.axis([min_x,max_x,min_y,max_y])
plt.xticks(remList)
plt.xlabel('Number of Feature Removals')
plt.ylabel('Accuracy')
plt.title('Accuracy Alteration')
plt.savefig('AccuracyAlteration.png')
plt.clf()


















