import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import operator
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.cross_validation import KFold
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix
from scipy import interp
from sklearn import svm
from random import *
from numpy.linalg import *
from time import *


def getNeighbors(train_set,test_in,K):
    distances = []
    for i,x in enumerate(train_set):
        dist = cosine(test_in, x)
        t = (i,dist)
        distances.append(t)
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x])
    return neighbors

def MajorityVote(neighbors,category_set):
    classVotes = {}
    for x in neighbors:
        index = x[0]
        response = category_set[index]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def KNN(train_set,category_set,test_set,K):
    class_list = []
    for c,instance in enumerate(test_set):
        neigh = getNeighbors(train_set,instance,K)
        class_ = MajorityVote(neigh,category_set)
        class_list.append(class_)
    return class_list


inputF = pd.read_csv('train_set.csv',sep='\t')
vectorizer = TfidfVectorizer(stop_words='english')
data = inputF['Title'] + inputF['Content']
vectorizer.fit_transform(data)
X_train = vectorizer.transform(data)
svd = TruncatedSVD(n_components=100)
X_lsi = svd.fit_transform(X_train)
categories = inputF['Category']
labelEnc = preprocessing.LabelEncoder()
labelEnc.fit(categories)
Y_train = labelEnc.transform(categories)

Y = label_binarize(Y_train, classes=[0, 1, 2, 3, 4])
svm_ = SVC(kernel = 'linear', probability = True, cache_size = 1000)
randFor_ = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
naiveB_ = MultinomialNB()


dictionary = {'SVM': svm_, 'Random Forest':randFor_, 'Naive Bayes': naiveB_, 'KNN':KNN}
names = {'Random Forest' : 'Random Forest', 'SVM' : 'Support Vector Machine','Naive Bayes' : 'Naive Bayes', 'KNN':'KNN'}
columns = ['SVM','Random Forest','Naive Bayes','KNN']
rows = ['Accuracy','Precision','Recall','F-Measure','AUC']
df_ = pd.DataFrame(index=rows,columns=columns)
df_ = df_.fillna(0.0)

num_of_splits = 10
kf = KFold(len(X_lsi),n_folds=num_of_splits)

points = 100 
sampling = 10 
mean_fpr = np.linspace(0,1,points)
mean_fpr = mean_fpr * mean_fpr * mean_fpr
mean_fpr = np.sqrt(mean_fpr)
best_acc = 0
best_cl = ' '
n_classes = 5
for classifier in dictionary:
    print ("Classifier %s starting" %classifier)
    totalAcc = 0
    totalPrec = 0
    totalRec = 0
    totalF = 0
    totalAUC = 0
    mean_tpr = {}
    for k in range(n_classes):
        mean_tpr[k] = 0.0   
    for train_index, test_index in kf:
        if classifier != 'Naive Bayes':
            X_train_counts = X_lsi[train_index]
            X_test_counts = X_lsi[test_index]
            Y_train_counts = Y_train[train_index]
            Y_test_counts = Y_train[test_index]
        else:
            X_train_counts = X_train[train_index]
            X_test_counts = X_train[test_index]
            Y_train_counts = Y_train[train_index]
            Y_test_counts = Y_train[test_index]
        if classifier == 'KNN':
            Y_pred = KNN(X_train_counts,Y_train_counts,X_test_counts,5)
        else:
            dictionary[classifier].fit(X_train_counts, Y_train_counts)
            Y_pred = dictionary[classifier].predict(X_test_counts)
        acc = accuracy_score(Y_test_counts,Y_pred)
        prec = precision_score(Y_test_counts,Y_pred,average="macro")
        recall = recall_score(Y_test_counts,Y_pred,average="macro")
        fMes = f1_score(Y_test_counts,Y_pred,average="macro")
        totalAcc += acc
        totalPrec += prec
        totalRec += recall
        totalF += fMes
        if classifier != 'KNN':
            for k in range(n_classes):
                y_train = Y[train_index,k]
                y_test = Y[test_index,k]
                prob = dictionary[classifier].fit(X_train_counts,y_train).predict_proba(X_test_counts)
                fpr, tpr,thresholds = roc_curve(y_test, prob[:,1])
                mean_tpr[k] += interp(mean_fpr,fpr,tpr)
    auc_res = []
    if classifier != 'KNN':
        for k in range(n_classes):
            mean_tpr[k] = mean_tpr[k]/len(kf)
            tpr = mean_tpr[k][0:points:sampling]
            tpr[0] = 0.0
            tpr[-1] = 1.0
            fpr = mean_fpr[0:points:sampling]
            fpr[0] = 0.0
            fpr[-1] = 1.0
            mean_auc = auc(fpr,tpr)
            plt.plot(fpr, tpr,label='Mean ROC of %s (area = %0.2f)' % (labelEnc.inverse_transform([k]),mean_auc), lw=2)
            auc_res += [mean_auc]
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Fold-Average ROC for each class')
        plt.legend(loc="lower right")
        plt.savefig(names[classifier] + '_rocPlot.png')
        plt.clf() 
    df_[classifier][0] = '%.3f' % (totalAcc/num_of_splits)
    df_[classifier][1] = '%.3f' % (totalPrec/num_of_splits)
    df_[classifier][2] = '%.3f' % (totalRec/num_of_splits)
    df_[classifier][3] = '%.3f' % (totalF/num_of_splits)
    df_[classifier][4] = '%.3f' % (sum(auc_res)/num_of_splits)
    if totalAcc > best_acc:
        best_acc = totalAcc
        best_cl = classifier
output = 'EvaluationMetric_10fold.csv'
df_.to_csv(output, sep='\t', encoding='utf-8')

TestF = pd.read_csv('test_set.csv',sep='\t')
Tvectorizer = TfidfVectorizer(stop_words='english')
Tdata = inputF['Title'] + inputF['Content']
Tvectorizer.fit_transform(Tdata)
X_test = vectorizer.transform(Tdata)
svd = TruncatedSVD(n_components=100)
TX_lsi = svd.fit_transform(X_test)
print ("Running %s on the test set." %best_cl)
if classifier != 'Naive Bayes' and classifier != 'KNN':
    dictionary[classifier].fit(X_lsi,Y_train)
    t_categories = dictionary[best_cl].predict(TX_lsi)
else:
    if classifier == 'KNN':
        t_categories = KNN(X_train,Y_train,X_test,5)
    else:
        dictionary[classifier].fit(X_train,Y_train)
        t_categories = dictionary[best_cl].predict(X_test)
pred_dict = {'ID':[],'Predicted Category':[]}
t_labels = labelEnc.inverse_transform(t_categories)
counter = 0
for c in t_labels:
    pred_dict['ID'].append(counter)
    pred_dict['Predicted Category'].append(c)
    counter += 1
test_df = pd.DataFrame(data = pred_dict)
test_df = test_df.ix[::, ['ID', 'Predicted Category']]
test_df.to_csv('testSet_categories.csv', sep = '\t', index = False) 
