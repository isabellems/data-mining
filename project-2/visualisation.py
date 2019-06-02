import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter


def hist(data1, data2, attrList, name):
    c1 = Counter(data1)
    c2 = Counter(data2)

    categories1 = []
    categories2 = []

    for a in attrList:
        categories1.append(c1[a])
        categories2.append(c2[a])

    bar_heights1 = tuple(categories1)
    bar_heights2 = tuple(categories2)
 
    fig, ax = plt.subplots()
 
    leng1 = len(bar_heights1)
    if (leng1 == 2) :
        x = (0.7 , 1.4)
        width = 0.6
        ax.set_xlim((0, 2.0))
    elif (leng1 == 3) :
        x = (0.7, 1.4, 2.1)
        width = 0.6
        ax.set_xlim((0, 2.5))
    elif (leng1 == 4) :
        x = (0.5, 1.0, 1.5, 2.0)
        width = 0.4
        ax.set_xlim((0, 2.5))
    elif (leng1 == 5) :
        x = (0.5, 1.0, 1.5, 2.0, 2.5)
        width = 0.4
        ax.set_xlim((0, 3.0))
    elif (leng1 == 10) :
        x = (0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0)
        width = 0.25
        ax.set_xlim((0, 4.4))
    elif (leng1 == 11) :
        x = (0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4)
        width = 0.25
        ax.set_xlim((0, 4.8))
 
    ax.bar(x, bar_heights1, width, color='blue', alpha=0.5, label='Good')
    ax.bar(x, bar_heights2, width, color='red', alpha=0.5, label = 'Bad')
    ax.legend(loc="upper right")
    ax.set_ylim((0, max(categories1 + categories2)*1.1))
    ax.set_xticks(x)
    ax.set_xticklabels(attrList)
    plt.title(name)
    plt.savefig('Plots/' + name + '.png')
    plt.clf()

def box(data1, data2, attrList, name):
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    data = [data1, data2]
    bp = ax.boxplot(data, patch_artist=True)
    bp['boxes'][0].set( color='red', alpha = 0.5, linewidth=2)
    bp['boxes'][1].set( color='blue', alpha = 0.5, linewidth=2)
    bp['medians'][0].set(color='blue', alpha = 0.5, linewidth=2)
    bp['medians'][1].set(color='red', alpha = 0.5, linewidth=2)
    for flier in bp['fliers']:
        flier.set(marker='o', color='red', linewidth=0.5)
    ax.set_xticklabels(['Good','Bad'])
    plt.title(name)
    fig.savefig('Plots/' + name + '.png', bbox_inches='tight')
    plt.clf()

inputF = pd.read_csv('train.tsv',sep='\t')
label = inputF["Label"]
good_boys = []
bad_boys = []

for y,col in enumerate(inputF):
    if col=='Label':
        break
    categories = set()
    for i,data in enumerate(inputF[col]):
      if(label[i] == 1):
          good_boys.append(data)
      else:
          bad_boys.append(data)
      categories = set(good_boys + bad_boys)
    num = y+1
    name = 'Attr' + str(num)
    if not str(good_boys[0]).isnumeric():
      hist(good_boys, bad_boys, sorted(categories),name)
    elif str(good_boys[0]).isnumeric():
      box(good_boys, bad_boys, sorted(categories),name) 
    del good_boys[:]
    del bad_boys[:]


