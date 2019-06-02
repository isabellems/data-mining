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
from sklearn.ensemble import VotingClassifier
from nltk.stem.snowball import SnowballStemmer
# from sklearn.cross_validation import cross_val_score
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix
from scipy import interp
from sklearn import svm
from random import *
from numpy.linalg import *
from time import *
import re

class Article:
    def __init__(self,title,content,category):
        self.title = title
        self.content = content
        self.category = category
    def tokenize(self):
        token = re.split(r'(?u)(?![\'])\W+',self.content)
        cont = ' '.join(token)
        self.tokContent = cont.split(' ')
        title = re.split(r'(?u)(?![\'])\W+',self.title)
        self.tokTitle = title
    def removeSW(self,SW):
        content = self.tokTitle + self.tokContent
        SWfree = [word for word in content if word.lower() not in SW]
        self.tokContent = ' '.join(SWfree)
    def getTitle(self):
        return self.title
    def getContent(self):
        return self.content
    def getCategory(self):
        return self.category
    def getTokContent(self):
        return self.tokContent
    def getTokTitle(self):
        return self.tokTitle


class Preprocessor:
    def __init__(self,input,stopw,cat):
        with open(stopw) as file:
            cont = file.readlines()
        self.stop_words = [x.strip() for x in cont] 
        self.set = input
        self.dataF = pd.read_csv(self.set, sep = '\t')
        self.titles = list(self.dataF['Title'])
        self.contents = list(self.dataF['Content'])
        self.articles = []
        self.numArts = len(self.titles)
        self.categorySet = []
        # self.stemmer = SnowballStemmer("english")
        if(cat != False):
            self.categories = list(self.dataF['Category'])
        for i in range(0, self.numArts):
            if(cat != False):
                article = Article(self.titles[i], self.contents[i], self.categories[i])
            else:
                article = Article(self.titles[i], self.contents[i])
            self.articles.append(article)
        catset = set(self.categories)
        self.categorySet = list(catset)
    def tokenize(self): 
        for i in range(0, self.numArts):
            self.articles[i].tokenize()
            self.articles[i].removeSW(self.stop_words)
    def getCategories(self):
        return self.categorySet
    def getCategoryList(self):
        return self.categories
    def rawData(self):
        articlesData = {'Title' : [], 'Content' : [], 'Category' : []}       
        for article in self.articles:
            articlesData['Title'].append(article.getTitle())
            articlesData['Content'].append(article.getTokContent())
            if self.categories != None:
                articlesData['Category'].append(article.getCategory())
        return pd.DataFrame(articlesData)



inputF = pd.read_csv('train_set.csv',sep='\t')
prepros = Preprocessor('train_set.csv','stopwords.txt',True)
prepros.tokenize()
data = prepros.rawData()
categories = prepros.getCategoryList()

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(data)
X_train = vectorizer.transform(data)
svd = TruncatedSVD(n_components=100)
X_lsi = svd.fit_transform(X_train)
labelEnc = preprocessing.LabelEncoder()
labelEnc.fit(categories)
Y_train = labelEnc.transform(categories)

Y = label_binarize(Y_train, classes=[0, 1, 2, 3, 4])
svm_ = SVC(kernel = 'linear', probability = True, cache_size = 1000)
randFor_ = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
naiveB_ = MultinomialNB()

dictionary = {'SVM': svm_, 'Random Forest':randFor_, 'Naive Bayes': naiveB_, 'KNN':KNN}

num_of_splits = 10
kf = KFold(len(X_lsi),n_folds=num_of_splits)

mjvClf = VotingClassifier(estimators = [('SVM', svm), ('Random Forest', randFor_), ('Naive Bayes', naiveB_)], voting='hard')

for train_index, test_index in kf:
    X_train_counts = X_train[train_index]
    X_test_counts = X_train[test_index]
    Y_train_counts = Y_train[train_index]
    Y_test_counts = Y_train[test_index]
    mjvClf.fit(X_train_counts, Y_train_counts)
    Y_pred = mjvClf.predict(X_test_counts)
    acc = accuracy_score(Y_test_counts,Y_pred)
    totalAcc += acc
    
    
newAcc = '%.3f' % (totalAcc/num_of_splits)
print ("Accuracy : %f" %newAcc)




