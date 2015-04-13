from collections import defaultdict
import csv
from csv import DictReader, DictWriter

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer
from nltk.util import ngrams

from sklearn.metrics import mean_squared_error

import wikipedia
import ast
import re
import math

def morphy_stem(word):
    """
    Simple stemmer
    """
    stem = wn.morphy(word)
    if stem:
        return stem.lower()
    # else:
    #     return word.lower()

def myround(x, base=5):
    return int(base * round(float(x)/base))

sign = lambda x: math.copysign(1, x)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample this amount')
    args = parser.parse_args()
    


    question_data = {}
    questions = DictReader(open("questions.csv"),'r')
    for ii in questions:
        # print ii['r']     #Question Number
        # print ii[None][0]     #Answer
        # print ii[None][2]       #Category
        # print ii[None][3]       #Question (all words)
        # print ii[None][4]       #Question (no stopwords)
        # print ii[None][5]       #Wiki article length (28000 avg, 0 was no article)

        # print ''
        question_data[int(ii['r'])] = [ii[None][0], ii[None][2], 
                                        ii[None][3], ii[None][4],
                                         ii[None][5]]


    # Try running on per user basis
    # Group users with <30 questions answered together
    user_dict = {}
    user_dict['bin1'] = []
    user_dict['bin2'] = []
    user_dict['bin3'] = []
    user_dict['bin4'] = []
    user_dict['bin5'] = []
    user_dict['bin6'] = []
    user_dict['bin7'] = []
    # user_dict['bin8'] = []

    user_stats = {}
    question_stats = {}

    train = DictReader(open("train3.csv", 'r'))
    for ii in train:
        question_stats[int(ii['question'])]= [0,0]

        if int(ii['QuestAnswered'])<=5:
            user_dict['bin1'].append(int(ii['user']))
        elif int(ii['QuestAnswered'])>5 and int(ii['QuestAnswered'])<=15:
            user_dict['bin2'].append(int(ii['user']))
        elif int(ii['QuestAnswered'])>15 and int(ii['QuestAnswered'])<=30:
            user_dict['bin3'].append(int(ii['user']))
        elif int(ii['QuestAnswered'])>30 and int(ii['QuestAnswered'])<=60:
            user_dict['bin4'].append(int(ii['user']))
        elif int(ii['QuestAnswered'])>60 and int(ii['QuestAnswered'])<=100:
            user_dict['bin5'].append(int(ii['user']))
        elif int(ii['QuestAnswered'])>100 and int(ii['QuestAnswered'])<=200:
            user_dict['bin6'].append(int(ii['user']))
        elif int(ii['QuestAnswered'])>200 and int(ii['QuestAnswered'])<=300:
            user_dict['bin7'].append(int(ii['user']))
        # else: 
            # user_dict['bin8'].append(int(ii['user']))
        else:
            user_dict[str(ii['user'])] = []
            user_dict[str(ii['user'])].append(int(ii['user']))

    overall_total = 0
    overall_correct = 0
    qans = 0
    


    for user in user_dict:    
        train = DictReader(open("train3.csv", 'r'))
        
        # Split off dev section
        dev_train = []
        dev_test = []
        answers = []
        qids = []


        for ii in train:

            # if int(ii['QuestAnswered'])<5:
            # if int(ii['user'])==148:
            if int(ii['user']) in user_dict[user]:
                qans = int(ii['QuestAnswered'])

                #Feature dictionary
                d = defaultdict(int)
                # d['Qid'] = ii['question']
                # d['user1'] = int(float(ii['user']))
                # d['userpos1'] = int(float(ii['AvgUserPos']))
                # d['qpos1'] = int(float(ii['AvgQuestPos']))
                # d['diff'] = round(abs(int(float(ii['AvgQuestPos']))) - abs(int(float(ii['AvgUserPos']))), -1)


                d['qpercent'] = myround(float(ii['QuestPercent'])*100, 25)
                d['upercent'] = myround(float(ii['UserPercent'])*100, 25)

                ##usually not useful
                # d['qans'] = int(float(ii['QuestAnswered']))
                # d['category1'] = question_data[int(ii['question'])][1]
                # d['wikilen1'] = int(question_data[int(ii['question'])][4])


                if int(ii['id']) % 5 == 0:
                    dev_test.append((d, sign(float(ii['position']))))
                    answers.append(float(ii['position']))
                    qids.append(int(ii['question']))

                else:
                    dev_train.append((d, sign(float(ii['position']))))
                    # dev_train.append((d, int(float(ii['RoundPos']))))


        # Train a classifier
        # print("Training classifier ...")
        # classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)
        classifier = nltk.classify.MaxentClassifier.train(dev_train, 'GIS', trace=0, max_iter=4)

        # classifier2.show_most_informative_features(20)

        predictions_sign = []
        correct = 0
        total = 0


        for ii in dev_test:
            # print classifier.classify(ii[0])
            ans = classifier.classify(ii[0])
            total+=1
            overall_total+=1
            question_stats[qids[total-1]][1]+=1
            if sign(ans) == sign(answers[total-1]):
                correct+=1
                overall_correct+=1
                question_stats[qids[total-1]][0]+=1


            # predictions_sign.append(1.0)

        # for ii in range(0,len(predictions_sign)):
        #     total+=1
        #     overall_total+=1
        #     if sign(predictions_sign[ii]) == sign(answers[ii]):
        #         correct+=1
        #         overall_correct+=1

        user_stats[user] = [round(float(correct)/total, 2), qans]
        # try:
        # print "Accuracy = ", round(float(correct)/total, 2), "User(s) = ", user
        # except ZeroDivisionError:
            # print "No questions in test? ", correct, total 


    for user in user_stats:
        if user_stats[user][0]<=1:
            print user, user_stats[user]

    print "Final Accuracy = ", float(overall_correct)/overall_total

    # print '## Question Data ##'
    # for qq in question_stats:
    #     print qq, round(question_stats[qq][0]/(question_stats[qq][1]+0.001,1)