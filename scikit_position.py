import csv
from csv import DictReader, DictWriter
from collections import defaultdict

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

#Round to nearest base
def myround(x, base=5):
    return int(base * round(float(x)/base))


# X_train_counts = count_vect.fit_transform(twenty_train.data)

question_data = {}
questions = DictReader(open("questions_processed.csv"),'r')
counter = 0
for ii in questions:
    # print ii
    # print ii['r']     #Question Number
    # print ii[None][0]     #Answer
    # print ii[None][2]       #Category
    # print ii[None][3]       #Question (all words)
    # print ii[None][4]       #Question (no stopwords)
    # print ii[None][5]       #Wiki article length (28000 avg, 0 was no article)
    if counter >0:
        question_data[int(float(ii['r']))] = [ii[None][0], ii[None][2], 
                                        ii[None][3], ii[None][4]]
    counter+=1
###question_data[id][0]=answer, [id][1]=caegory
### [id][2]=question and POS, [id][3]=question(no stopwords)



train = DictReader(open("train3.csv", 'r'))


#Count Vectorize category
vocab = set()
pos_tags = set()
words = set()
for ii in train:
    vocab.add(question_data[int(ii['question'])][1])
    for jj in eval(question_data[int(ii['question'])][2]):
        pos_tags.add(jj[1])
        words.add(jj[0])
vocab = list(vocab)
pos_tags = list(pos_tags)

cv_category = CountVectorizer(vocabulary=vocab)
category_counts = cv_category.fit_transform(vocab)
# cat = cv_category.vocabulary_.get('Fine Arts')

cv_pos = CountVectorizer(vocabulary=pos_tags)
pos_counts = cv_pos.fit_transform(pos_tags)
# tag = cv_pos.vocabulary_.get('NP')
# print tag

cv_words = CountVectorizer(vocabulary=words)
words_counts = cv_words.fit_transform(words)

train = DictReader(open("train3.csv", 'r'))
dev_train_feats=[]
dev_test_feats=[]
dev_train_labels=[]
dev_test=[]

for ii in train:
    userpos = myround(int(float(ii['AvgUserPos'])),25)
    questpos = myround(int(float(ii['AvgQuestPos'])),25)
    diff = myround(int(float(ii['AvgQuestPos'])) - int(float(ii['AvgUserPos'])),10)
    cat = cv_category.vocabulary_.get(question_data[int(ii['question'])][1])
    numans = myround(int(float(ii['QuestAnswered'])),25)
    qper = myround(float(ii['QuestPercent']),.1)
    uper = myround(float(ii['UserPercent']),.1)

    # NPs = 0
    # NNs = 0
    # this = 0
    # dates = 0
    # counter = 0
    # for jj in eval(question_data[int(ii['question'])][2]):
    #     if counter == 0:
    #         if jj[0]=='This':
    #             this = 1
    #         firsttag = cv_pos.vocabulary_.get(jj[1])
    #         firstword = cv_words.vocabulary_.get(jj[0])
    #     if counter<userpos:
    #         if jj[1]=='NP' or jj[1]=='NP-TL':
    #             NPs += 1
    #         if jj[1]=='NN' or jj[1]=='NN-TL':
    #             NNs += 1
    #         if jj[1]=='CD':
    #             dates += 1
    #     counter += 1

    features = [userpos,questpos,diff,cat,numans,qper,uper]

    if (int(ii['id']))%5==0:
        dev_test.append(ii)
        dev_test_feats.append(features)

    else:
        dev_train_feats.append(features)
        dev_train_labels.append(int(ii['position']))


x_train = dev_train_feats
x_test = dev_test_feats
y_train = dev_train_labels
        


# Train classifier
print "Training Classifier"
# lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
# lr.fit(x_train, y_train)

# regr = linear_model.LinearRegression()
# regr = linear_model.BayesianRidge()
regr = ensemble.GradientBoostingRegressor(learning_rate=0.5, n_estimators=200, max_depth=3)
# regr = ensemble.AdaBoostRegressor()
regr.fit(x_train, y_train)


# predictions = lr.predict(x_test)
predictions = regr.predict(x_test)
dict1={}
for ii, pp in zip([x['id'] for x in dev_test], predictions):
    # print int(ii), ii, pp
    dict1[int(ii)]=int(pp)

right=0
total=len(dev_test)

answers = []
for ii in dev_test:
    try:
    	answers.append(int(ii['position']))
    except KeyError:
        print ii['id'],ii['position']

average = np.mean(answers)
mm = []
for ii in answers:
    mm.append(average)


# print predictions
# print answers

rms = mean_squared_error(predictions, answers) ** 0.5
print "RMS Accuracy Position Only = ", rms

##########################################################
##########################################################
##### Run again for all data and output a submission #####
##########################################################
##########################################################



## Start training


train = DictReader(open("train3.csv", 'r'))
dev_train_feats=[]
dev_test_feats=[]
dev_train_labels=[]
dev_test=[]

for ii in train:
    userpos = int(float(ii['AvgUserPos']))
    questpos = int(float(ii['AvgQuestPos']))
    diff = int(float(ii['AvgQuestPos'])) - int(float(ii['AvgUserPos']))
    cat = cv_category.vocabulary_.get(question_data[int(ii['question'])][1])
    numans = int(float(ii['QuestAnswered']))
    qper = float(ii['QuestPercent'])
    uper = float(ii['UserPercent'])

    features = [userpos,questpos,diff,cat,numans,qper,uper]

    dev_train_feats.append(features)
    dev_train_labels.append(int(ii['position']))


x_train = dev_train_feats
y_train = dev_train_labels
        

# Train classifier
print "Training classifier on all data"
# lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
# lr.fit(x_train, y_train)

# regr = linear_model.LinearRegression()
# regr = linear_model.BayesianRidge()
regr = ensemble.GradientBoostingRegressor(learning_rate=0.5, n_estimators=200, max_depth=3)
# regr = ensemble.AdaBoostRegressor()
regr.fit(x_train, y_train)

x_test = []

counter = 0
with open('test2.csv') as f:
    test = csv.reader(f)
    for row in test:
        if counter==0:
            print row
        if counter>0:
            userpos = int(float(row[3]))
            questpos = int(float(row[4]))
            diff = questpos-userpos
            cat = cv_category.vocabulary_.get(row[6])
            numans = int(row[5])
            qper = float(row[8])
            uper = float(row[7])
        # try:
        #     qper = data_q[int(ii['question'])][0][1]
        # except KeyError:
        #     qper = one_q_per

            features = [userpos,questpos,diff,cat,numans,qper,uper]
            # print features
            x_test.append(features)
        counter+=1
   
 

predictions = regr.predict(x_test)


# for ii in predictions:
#     print ii

o = DictWriter(open('pred2.csv', 'w'), ['id', 'position'])
o.writeheader()

counter = 0
with open('test2.csv') as f:
    test = csv.reader(f)
    for row in test:
        # if counter == 0:
        #     print row

        if counter>0:

            o.writerow({'id': row[0], 'position': predictions[counter-1]})
            # o.writerow({'id': row[0], 'position': 1})
        counter+=1

# test = DictReader(open('test.csv'),'r')

# o = DictWriter(open('pred2.csv', 'w'), ['id', 'position'])
# o.writeheader()
# for ii in test:
#     o.writerow({'id': ii['id'], 'position': final_answers[ii]})


# # test = DictReader(open('test.csv'),'r')
# for ii in test:
#     print ii
#     print int(ii['user'])
#     userpos = alldata[int(ii['user'])][0][0]
#     try:
#         userpos = alldata[int(ii['user'])][0][0]
#     except KeyError:
#         userpos = one_q_user_pos

#     try:
#         questpos = data_q[int(ii['question'])][0][0]
#         diff = alldata[int(ii['user'])][0][0] - data_q[int(ii['question'])][0][0]
#     except KeyError:
#         questpos = one_q_pos
#         diff = one_q_user_diff

#     try:
#         cat = cv_category.vocabulary_.get(question_data[int(ii['question'])][1])
#     except:
#         cat = 4

#     try:
#         numans = alldata[int(ii['user'])][0][6]
#     except KeyError:
#         numans = 1

#     # try:
#     #     qper = data_q[int(ii['question'])][0][1]
#     # except KeyError:
#     #     qper = one_q_per

#     features = [userpos,questpos,diff,cat,numans]
#     print features
#     dev_test.append(features)

   
# classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test)