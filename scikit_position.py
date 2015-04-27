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
	diff = int(float(ii['AvgQuestPos'])) - int(float(ii['AvgUserPos']))
	cat = cv_category.vocabulary_.get(question_data[int(ii['question'])][1])
	numans = int(float(ii['QuestAnswered']))
	qper = float(ii['QuestPercent'])
	# uper = float(ii['UserPercent'])

	features = [userpos,questpos,diff,cat,numans]

	if int(ii['id'])%5==0:
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

# print predictions
# print answers

rms = mean_squared_error(predictions, answers) ** 0.5
print "RMS Accuracy Position Only = ", rms

##########################################################
##########################################################
##### Run again for all data and output a submission #####
##########################################################
##########################################################

#Stuff for unseen questions/users
train = DictReader(open("train3.csv", 'r'))


alldata = {}
data_q = {}
data_one = {}
train_users = set()

for ii in train:
    if int(ii['QuestAnswered'])<2:
        data_one[int(ii['user'])] = []
    alldata[int(ii['user'])] = []
    data_q[int(ii['question'])] = []
    train_users.add(int(ii['user']))

train = DictReader(open("train3.csv", 'r'))
for ii in train:

    d = defaultdict(int)
    # d['Qid'] = ii['question']
    # d['user1'] = int(float(ii['user']))
    d['userpos1'] = abs(int(float(ii['AvgUserPos'])))
    d['qpos1'] = abs(int(float(ii['AvgQuestPos'])))
    d['diff'] = abs(int(float(ii['AvgQuestPos']))) - abs(int(float(ii['AvgUserPos'])))
    qper = myround(float(ii['QuestPercent'])*100, 25)
    uper = myround(float(ii['UserPercent'])*100, 25)
    d['category1'] = question_data[int(ii['question'])][1]

    if int(ii['QuestAnswered'])<2:
        data_one[int(ii['user'])].append([d['userpos1'], d['qpos1'], d['diff'], qper, uper, d['category1'], ii['QuestAnswered']])
        # print data_one[int(ii['user'])]

    alldata[int(ii['user'])].append([d['userpos1'], d['qpos1'], d['diff'], qper, uper, d['category1'], ii['QuestAnswered']])
    data_q[int(ii['question'])].append([d['qpos1'], qper])

# for never seen user    
one_q_user_pos = []
one_q_user_qpos = []
one_q_user_diff = []
one_q_user_qper = []
one_q_user_uper = []

for user in data_one:
    # print "dat[user] = ,", data_one[user]
    for row in data_one[user]:
        # print "user, ROW = ", user,row
        one_q_user_pos.append(int(float(row[0])))
        one_q_user_qpos.append(int(float(row[1])))
        one_q_user_diff.append(int(float(row[2])))
        one_q_user_qper.append(float(row[3]))
        one_q_user_uper.append(float(row[4]))

# print one_q_user_uper

one_q_user_pos = int(np.mean(one_q_user_pos))
one_q_user_uper = np.mean(one_q_user_uper)
one_q_user_qper = np.mean(one_q_user_qper)
one_q_user_qpos = int(np.mean(one_q_user_qpos))
one_q_user_diff = int(np.mean(one_q_user_diff))

one_q_per = []
one_q_pos = []

## for never seen questions
for quest in data_q:
    for row in data_q[quest]:
        one_q_per.append(float(row[1]))
        one_q_pos.append(int(float(row[0])))

one_q_per = np.mean(one_q_per)
one_q_pos = int(np.mean(one_q_pos))

# print one_q_per
# print one_q_pos

# print one_q_user_uper
# print one_q_user_pos


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
    # uper = float(ii['UserPercent'])

    features = [userpos,questpos,diff,cat,numans]

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

counter = 0
with open('test.csv') as f:
    test = csv.reader(f)
    for row in test:
        if counter>0:
            try:
                userpos = alldata[int(row[2])][0][0]
            except KeyError:
                userpos = one_q_user_pos

            try:
                questpos = data_q[int(row[1])][0][0]
                diff = alldata[int(row[2])][0][0] - data_q[int(row[1])][0][0]
            except KeyError:
                questpos = one_q_pos
                diff = one_q_user_diff

            try:
                cat = cv_category.vocabulary_.get(question_data[int(row[1])][1])
            except:
                cat = 4

            try:
                numans = alldata[int(row[2])][0][6]
            except KeyError:
                numans = 1

        # try:
        #     qper = data_q[int(ii['question'])][0][1]
        # except KeyError:
        #     qper = one_q_per

            features = [userpos,questpos,diff,cat,numans]
            # print features
            dev_test.append(features)
        counter+=1
   
 
x_test = dev_test
predictions = regr.predict(x_test)

# for ii in predictions:
#     print ii

o = DictWriter(open('pred2.csv', 'w'), ['id', 'position'])
o.writeheader()

counter = 0
with open('test.csv') as f:
    test = csv.reader(f)
    for row in test:
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