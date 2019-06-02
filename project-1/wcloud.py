from os import path
from scipy.misc import imread
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re
import time

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
		self.tokContent = SWfree
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
	def getCategoryContent(self,category):
		contList = []
		sep = " "
		for i in range(0, self.numArts):
			if self.articles[i].getCategory() is category:
				con = self.articles[i].getTokContent()
				contList = contList + con

		contents = sep.join(contList)
		return contents



print ("Preprocessing data...")
prepros = Preprocessor('train_set.csv','stopwords.txt',True)
print ("Tokenizing content...")
prepros.tokenize()
categories = prepros.getCategories()
for category in categories:
	print ("Creating Wordcloud for %s" % category)
	content = prepros.getCategoryContent(category)
	wordcl = WordCloud(max_words=300,max_font_size=50, margin=10, random_state=1, width=840, height=420).generate(content)
	output = category + ".png"
	wordcl.to_file(output)

