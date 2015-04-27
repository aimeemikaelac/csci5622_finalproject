from csv import DictReader, DictWriter

import numpy as np
from numpy import array


from sklearn import datasets, linear_model, ensemble
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer
from nltk.util import ngrams
from nltk.corpus import stopwords

import re
import math

#Round to nearest base
def myround(x, base=5):
    return int(base * round(float(x)/base))

sign = lambda x: math.copysign(1, x)

# X_train_counts = count_vect.fit_transform(twenty_train.data)

question_data = {}
questions = DictReader(open("questions.csv"),'r')
for ii in questions:
    # print ii['r']     #Question Number
    # print ii[None][0]     #Answer
    # print ii[None][2]       #Category
    # print ii[None][3]       #Question (all words)
    # print ii[None][4]       #Question (no stopwords)
    # print ii[None][5]       #Wiki article length (28000 avg, 0 was no article)
    question_data[int(ii['r'])] = [ii[None][0], ii[None][2], 
                                    ii[None][3], ii[None][4],
                                     ii[None][5]]
###question_data[id][0]=answer, [id][1]=caegory
### [id][2]=question, [id][3]=question(no stopwords)
### [id][4]=wikilength



train = DictReader(open("train3.csv", 'r'))


#Count Vectorize category
vocab = set()
for ii in train:
	vocab.add(question_data[int(ii['question'])][1])
vocab = list(vocab)

cv_category = CountVectorizer(vocabulary=vocab)
category_counts = cv_category.fit_transform(vocab)
cat = cv_category.vocabulary_.get('Fine Arts')




train = DictReader(open("train3.csv", 'r'))
dev_train_feats=[]
dev_test_feats=[]
dev_train_labels=[]
dev_test=[]

for ii in train:
    userpos = int(float(ii['AvgUserPos']))
    questpos = int(float(ii['AvgQuestPos']))
    cat = cv_category.vocabulary_.get(question_data[int(ii['question'])][1])
    numans = int(float(ii['QuestAnswered']))

    # features = [userpos,questpos,diff,cat,numans]

    features = [userpos,questpos,cat,numans]

    if int(ii['id'])%5==0:
        dev_test.append(ii)
        dev_test_feats.append(features)

    else:
        dev_train_feats.append(features)
        dev_train_labels.append(sign(int(ii['position'])))


x_train = dev_train_feats
x_test = dev_test_feats
y_train = dev_train_labels
        


# Train classifier
print "Training Classifier"
# lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
lr = ensemble.GradientBoostingClassifier()
lr.fit(x_train, y_train)

# regr = linear_model.LinearRegression()
# regr = linear_model.BayesianRidge()
# regr = ensemble.GradientBoostingRegressor(learning_rate=0.5, n_estimators=200, max_depth=3)
# regr = ensemble.AdaBoostRegressor()
# regr.fit(x_train, y_train)


predictions = lr.predict(x_test)
# predictions = regr.predict(x_test)
dict1={}
for ii, pp in zip([x['id'] for x in dev_test], predictions):
    # print int(ii), ii, pp
    dict1[int(ii)]=sign(pp)

right=0
total=len(dev_test)

answers = []
for ii in dev_test:
    try:
        if sign(int(ii['position']))==dict1[int(ii['id'])]:
    	   right+=1
    except KeyError:
        print ii['id'],ii['position']


print("Accuracy on dev: %f" % (float(right) / float(total)))