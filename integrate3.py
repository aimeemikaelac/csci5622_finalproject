import csv
from csv import DictReader, DictWriter
from collections import defaultdict

import numpy as np
from numpy import array

from sklearn import datasets, linear_model, ensemble
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

import re
import string

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer
from nltk.util import ngrams
from nltk.corpus import stopwords
kTOKENIZER = TreebankWordTokenizer()
stop = stopwords.words('english')


#Train a classifier to predict question position and question percentage based on other questions
#Use to fill in test2.csv with predicted percentage/position.
#Also all columns for other features to make coding easier...

#Round to nearest base
def myround(x, base=5):
    return int(base * round(float(x)/base))

class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(ngram_range=(1, 2),analyzer=self.words_and_chars)

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            top10 = np.argsort(classifier.coef_[i])[-10:]
            print("%s: %s" % (category, " ".join(feature_names[top10])))


    def words_and_chars(self,text):
        words=re.findall(r'\w{3,}', text)
        for w in words:
            if w not in stop:
                # stem = wn.morphy(w)
                # if stem:
                #     yield stem.lower()
                yield w
                ngr=3
                if len(w)>ngr:
                    for i in range(len(w)-ngr+1):
                        yield w[i:i+ngr]


###############################################################################################################
###############################################################################################################
#Stuff for unseen questions/users
train = DictReader(open("train3.csv", 'r'))


alldata = {}
data_q = {}
data_one = {}
dev_train_feats=[]
dev_test_feats=[]
dev_train_labels=[]
dev_test=[]
train_users = set()

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

# for ii in train:
#     if int(ii['QuestAnswered'])<2:
#         data_one[int(ii['user'])] = []
#     alldata[int(ii['user'])] = []
#     data_q[int(ii['question'])] = []
#     train_users.add(int(ii['user']))

# train = DictReader(open("train3.csv", 'r'))
# for ii in train:

#     d = defaultdict(int)
#     # d['Qid'] = ii['question']
#     # d['user1'] = int(float(ii['user']))
#     d['userpos1'] = abs(int(float(ii['AvgUserPos'])))
#     d['qpos1'] = abs(int(float(ii['AvgQuestPos'])))
#     d['diff'] = abs(int(float(ii['AvgQuestPos']))) - abs(int(float(ii['AvgUserPos'])))
#     qper = float(ii['QuestPercent'])
#     uper = float(ii['UserPercent'])
#     d['category1'] = question_data[int(ii['question'])][1]

#     if int(ii['QuestAnswered'])<2:
#         data_one[int(ii['user'])].append([d['userpos1'], d['qpos1'], d['diff'], qper, uper, d['category1'], ii['QuestAnswered']])
#         # print data_one[int(ii['user'])]

#     alldata[int(ii['user'])].append([d['userpos1'], d['qpos1'], d['diff'], qper, uper, d['category1'], ii['QuestAnswered']])
#     data_q[int(ii['question'])].append([d['qpos1'], qper, d['category1']])

# # for never seen user    
# one_q_user_pos = []
# one_q_user_qpos = []
# one_q_user_diff = []
# one_q_user_qper = []
# one_q_user_uper = []

# for user in data_one:
#     # print "dat[user] = ,", data_one[user]
#     for row in data_one[user]:
#         # print "user, ROW = ", user,row
#         one_q_user_pos.append(int(float(row[0])))
#         one_q_user_qpos.append(int(float(row[1])))
#         one_q_user_diff.append(int(float(row[2])))
#         one_q_user_qper.append(float(row[3]))
#         one_q_user_uper.append(float(row[4]))

# # print one_q_user_uper

# one_q_user_pos = int(np.mean(one_q_user_pos))
# one_q_user_uper = np.mean(one_q_user_uper)
# one_q_user_qper = np.mean(one_q_user_qper)
# one_q_user_qpos = int(np.mean(one_q_user_qpos))
# one_q_user_diff = int(np.mean(one_q_user_diff))

# one_q_per = []
# one_q_pos = []


# unseen_questions = []
# unseen_users = []




######################################################################################33
########################################################################################

# train = DictReader(open("train3.csv", 'r'))
#Count Vectorize category
# vocab = set()
# vocab2 = set()
# for ii in train:
#     vocab.add(question_data[int(ii['question'])][1])
#     for jj in question_data[int(ii['question'])][2].split(' '):
#         # jj = re.sub("'s","",jj)
#         jj = jj.strip(string.punctuation)
#         # print jj
#         vocab2.add(jj)

# vocab = list(vocab)
# vocab2 = list(vocab2)
# print vocab2

# cv_category = CountVectorizer(vocabulary=vocab)
# category_counts = cv_category.fit_transform(vocab)
# cat = cv_category.vocabulary_.get('Fine Arts')

# cv_words = CountVectorizer(vocabulary=vocab2,ngram_range=(1, 2),analyzer=words_and_chars)
# cv_words = CountVectorizer(vocabulary=vocab2,ngram_range=(1, 2))
# word_counts = cv_words.fit_transform(vocab2)
# ww = cv_words.vocabulary_.get('this')

# print ww
# feat = Featurizer()


############################################################################################
# Classifier for average question position

train = DictReader(open("train3.csv", 'r'))
dev_train_feats=[]
dev_test_feats=[]
dev_train_labels=[]
dev_test=[]

for ii in train:
    # # cat = cv_category.vocabulary_.get(question_data[int(ii['question'])][1])
    # # cat = question_data[int(ii['question'])][1]
    # # length = len(question_data[int(ii['question'])][2])
    # unigrams = ''
    # # unigrams = []
    # for jj in question_data[int(ii['question'])][2].split(' '):
    #     jj = jj.strip(string.punctuation)
    #     unigrams+=' '+jj

    # features = unigrams
    # # features = [cat, length, unigrams]

    t=''
    for word in kTOKENIZER.tokenize(question_data[int(ii['question'])][2].strip(string.punctuation)):
        if word not in stop:
            t+=' '+ word
            # print word
  

        # dev_test_feats.append(t)

    if int(ii['question'])%5==0:
        dev_test.append(ii)
        dev_test_feats.append(t)

    else:
        dev_train_feats.append(t)
        dev_train_labels.append(int(float(ii['AvgQuestPos'])))

feat = Featurizer()

x_train = feat.train_feature(dev_train_feats)
x_test = feat.test_feature(dev_test_feats)
y_train = array(dev_train_labels)

# x_train = feat.train_feature(dev_train_feats).toarray()
# # x_train = x_train.toarray()
# # x_train = dev_train_feats
# x_test = feat.test_feature(dev_test_feats).toarray()
# y_train = array(dev_train_labels)
        
# print np.asarray(x_train).dtype
# print np.asarray(y_train).dtype
# print np.unique(map(len, x_train))
# print np.unique(map(len, y_train))

# Train classifier
print "Training Classifier"
# lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
# lr.fit(x_train, y_train)
# predictions = lr.predict(x_test)

regr = ensemble.GradientBoostingRegressor(learning_rate=0.5, n_estimators=200, max_depth=3)
regr.fit(x_train, y_train)
predictions = regr.predict(x_test)

# dict1={}
# for ii, pp in zip([x['id'] for x in dev_test], predictions):
#     # print int(ii), ii, pp
#     dict1[int(ii)]=int(pp)

# right=0
# total=len(dev_test)

# answers = []
# for ii in dev_test:
#     try:
#         answers.append(int(float(ii['AvgQuestPos'])))
#     except KeyError:
#         print ii['id'],ii['AvgQuestPos']

# # print predictions
# # print answers

# rms = mean_squared_error(predictions, answers) ** 0.5
# print "RMS Accuracy Position Only = ", rms


#########################################################################################3
########################################################################################


# with open('test.csv', 'rb') as test, open('test2.csv', 'wb') as test2:
#     csvreader = csv.DictReader(test)
#     fieldnames = csvreader.fieldnames + ['AvgUserPos'] + ['AvgQuestPos'] + \
#                                         ['QuestAnswered'] + ['Cat'] + \
#                                         ['UserPercent'] + ['QuestPercent'] + \
#                                         ['Answer'] + ['Words'] + \
#     print fieldnames
#     csvwriter = csv.DictWriter(test2, fieldnames)
#     csvwriter.writeheader()
#     for row in csvreader:
#         try:
#             row['AvgUserPos'] = alldata[int(row['user'])][0][0]
#         except KeyError:
#             row['AvgUserPos'] = one_q_user_pos
#         try:
#             row['AvgQuestPos'] = data_q[int(row['question'])][0][0]
#         except KeyError:
#             row['AvgQuestPos'] = 999999
#         try:
#             row['QuestAnswered'] = alldata[int(row['user'])][0][6]
#         except KeyError:
#             row['QuestAnswered'] = 1
#         try:
#             row['UserPercent'] = alldata[int(row['user'])][0][4]
#         except KeyError:
#             row['UserPercent'] = one_q_user_uper
#         try:
#             row['QuestPercent'] = data_q[int(row['question'])][0][1]
#         except KeyError:
#             row['QuestPercent'] = 999999
        
#         try:
#             row['Cat'] = question_data[int(row['question'])][1]
#             row['Answer'] = question_data[int(row['question'])][0]
#             row['Words'] = question_data[int(row['question'])][3]
#         except KeyError:
#             print int(row['id'])

#         # print row
#         csvwriter.writerow(row)