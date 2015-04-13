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
import numpy as np

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

    ###question_data[id][0]=answer, [id][1]=caegory
    ### [id][2]=question, [id][3]=question(no stopwords)
    ### [id][4]=wikilength

    # Read in training data
    train = DictReader(open("train3.csv", 'r'))
    
    # Split off dev section 80/20
    dev_train = []
    dev_test = []
    answers = []
    qids = []
    position_answers = {}
    final_answers = {}

    alldata = {}

    for ii in train:


        #Feature dictionary
        d = defaultdict(int)
        # d['Qid'] = ii['question']
        # d['user1'] = int(float(ii['user']))
        d['userpos1'] = abs(int(float(ii['AvgUserPos'])))
        d['qpos1'] = abs(int(float(ii['AvgQuestPos'])))
        d['diff'] = abs(int(float(ii['AvgQuestPos']))) - abs(int(float(ii['AvgUserPos'])))
               
        # d['qpercent'] = myround(float(ii['QuestPercent'])*100, 25)
        d['upercent'] = myround(float(ii['UserPercent'])*100, 25)

        ##usually not useful
        # d['category1'] = question_data[int(ii['question'])][1]

        alldata[int(ii['user'])] = [d['userpos1'], d['qpos1'], d['diff'], d['qpercent'], d['upercent'], d['category1'], ii['QuestAnswered']]
        # d['qans'] = int(float(ii['QuestAnswered']))
        # d['wikilen1'] = int(question_data[int(ii['question'])][4])


        dev_train.append((d, myround(abs(int(float(ii['position']))), 5)))


    #Make stats in case a user or question doesn't exist:
    one_q_user_pos = []
    one_q_user_qpos = []
    one_q_user_diff = []
    one_q_user_qper = []
    one_q_user_uper = []
    

    for user in alldata:
        one_q_user_pos.append(int(float(alldata[user][0])))
        one_q_user_qpos.append(int(float(alldata[user][1])))
        one_q_user_diff.append(int(float(alldata[user][2])))
        one_q_user_qper.append(float(alldata[user][3]))
        one_q_user_uper.append(float(alldata[user][4]))

    one_q_user_pos = np.mean(one_q_user_pos)
    one_q_user_uper = np.mean(one_q_user_uper)
    one_q_user_qper = np.mean(one_q_user_qper)
    one_q_user_qpos = np.mean(one_q_user_qpos)
    one_q_user_diff = np.mean(one_q_user_diff)
        


    # Train a classifier
    print("Training position classifier ...")
    # classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)
    classifier = nltk.classify.MaxentClassifier.train(dev_train, 'GIS', trace=3, max_iter=5)


    full_test = []

    test = DictReader(open("test.csv", 'r'))
    for ii in test:
        d = defaultdict(int)
        try:
            d['userpos1'] = alldata[int(ii['user'])][0]
            d['qpos1'] = alldata[int(ii['user'])][1]
            d['diff'] = alldata[int(ii['user'])][2]
            d['upercent'] = alldata[int(ii['user'])][4]
        except KeyError:
            d['userpos1'] = one_q_user_qpos
            d['qpos1'] = one_q_user_qpos
            d['diff'] = one_q_user_diff
            d['upercent'] = one_q_user_uper
        full_test.append((d, int(ii['id'])))

    for ii in full_test:
        # position answers[id] = position...
        position_answers[ii[1]] = classifier.classify(ii[0])
        # print position_answers[ii[1]]
    # print "Length first full test = ", len(full_test)

####################################################
## Correctness Section #############################
####################################################


    
    # Try running on per user basis
    # Group users with ??? questions answered together
    user_dict = {}
    user_dict['bin1'] = set()
    user_dict['bin2'] = set()
    user_dict['bin3'] = set()
    user_dict['bin4'] = set()
    user_dict['bin5'] = set()
    user_dict['bin6'] = set()
    user_dict['bin7'] = set()


    train = DictReader(open("train3.csv", 'r'))
    for ii in train:

        if int(ii['QuestAnswered'])<=5:
            user_dict['bin1'].add(int(ii['user']))
        elif int(ii['QuestAnswered'])>5 and int(ii['QuestAnswered'])<=15:
            user_dict['bin2'].add(int(ii['user']))
        elif int(ii['QuestAnswered'])>15 and int(ii['QuestAnswered'])<=30:
            user_dict['bin3'].add(int(ii['user']))
        elif int(ii['QuestAnswered'])>30 and int(ii['QuestAnswered'])<=60:
            user_dict['bin4'].add(int(ii['user']))
        elif int(ii['QuestAnswered'])>60 and int(ii['QuestAnswered'])<=100:
            user_dict['bin5'].add(int(ii['user']))
        elif int(ii['QuestAnswered'])>100 and int(ii['QuestAnswered'])<=200:
            user_dict['bin6'].add(int(ii['user']))
        elif int(ii['QuestAnswered'])>200 and int(ii['QuestAnswered'])<=300:
            user_dict['bin7'].add(int(ii['user']))
        else:
            user_dict[str(ii['user'])] = []
            user_dict[str(ii['user'])].append(int(ii['user']))

    for user in user_dict:    
        train = DictReader(open("train3.csv", 'r'))
        dev_train = []        

        for ii in train:
            #Feature dictionary
            if int(ii['user']) in user_dict[user]:
                d = defaultdict(int)
                d['qpercent'] = myround(float(ii['QuestPercent'])*100, 25)
                d['upercent'] = myround(float(ii['UserPercent'])*100, 25)
                # print sign(float(ii['position']))
                dev_train.append((d, sign(float(ii['position']))))

        # Train a classifier
        # print("Training classifier ...")
        # classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)
        classifier = nltk.classify.MaxentClassifier.train(dev_train, 'GIS', trace=0, max_iter=4)

       
    #make a list of all users in test set
    test_users_dict = {}
    # test_users_dict = {}
    test_users_dict['bin1'] = set()
    test_users_dict['bin2'] = set()
    test_users_dict['bin3'] = set()
    test_users_dict['bin4'] = set()
    test_users_dict['bin5'] = set()
    test_users_dict['bin6'] = set()
    test_users_dict['bin7'] = set()
    # test_users_list = []
    test = DictReader(open("test.csv", 'r'))
    for ii in test:
        # test_users_list.append(int(ii['user']))
        try:
            # print int(alldata[int(ii['user'])][6])
            qanswered = int(alldata[int(ii['user'])][6])
            if qanswered<=5:
                test_users_dict['bin1'].add(int(ii['user']))
            elif qanswered>5 and qanswered<=15:
                test_users_dict['bin2'].add(int(ii['user']))
            elif qanswered>15 and qanswered<=30:
                test_users_dict['bin3'].add(int(ii['user']))
            elif qanswered>30 and qanswered<=60:
                test_users_dict['bin4'].add(int(ii['user']))
            elif qanswered>60 and qanswered<=100:
                test_users_dict['bin5'].add(int(ii['user']))
            elif qanswered>100 and qanswered<=200:
                test_users_dict['bin6'].add(int(ii['user']))
            elif qanswered>200 and qanswered<=300:
                test_users_dict['bin7'].add(int(ii['user']))
            else:
                test_users_dict[str(ii['user'])] = []
                test_users_dict[str(ii['user'])].append(int(ii['user']))
        #never before seen user
        except KeyError:
            test_users_dict['bin1'].add(int(ii['user']))
            # pass

    # print test_users_dict['bin1']
    # print test_users_dict['bin2']
    # print test_users_dict['bin3']
    # print test_users_dict['bin4']
    # print test_users_dict['bin5']
    # print test_users_dict['bin6']
    # print test_users_dict['bin7']

    sign_answers = {}
    full_test = []


    for user in test_users_dict:    
        test = DictReader(open("test.csv", 'r'))
        counter = 0
        for ii in test:
            counter +=1
            #Feature dictionary
            d = defaultdict(int)
            #user seen in training

            if int(ii['user']) in test_users_dict[user]:
                try:
                    d['qpercent'] = alldata[int(ii['user'])][3]
                    d['upercent'] = alldata[int(ii['user'])][4]
                #never seen before user
                except KeyError:
                    d['qpercent'] = one_q_user_qper
                    d['upercent'] = one_q_user_uper
                full_test.append((d, int(ii['id'])))

            
        # print counter

    #classify position
    for ii in full_test:
        #sign_answers[id] = answer
        sign_answers[ii[1]] = classifier.classify(ii[0])
        # print sign_answers[ii[1]]
    # print "Length second full test = ", len(full_test)

    final_answers = {}
    for ii in sign_answers:
        final_answers[ii] = sign_answers[ii]*position_answers[ii]
        print final_answers[ii], sign_answers[ii], position_answers[ii]


    # Write predictions
    o = DictWriter(open('pred.csv', 'w'), ['id', 'position'])
    o.writeheader()
    for ii in final_answers:
        o.writerow({'id': ii, 'position': final_answers[ii]})

    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test)
    classifier.show_most_informative_features(20)