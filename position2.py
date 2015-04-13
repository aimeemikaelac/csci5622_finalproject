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
    
    # questions = {}
    # with open('questions.csv') as q:
    #     for ii in q:
    #         print ii
    #         # jj = ii.split('\n')
    #         # print jj[0]
    # q.close()

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

    # datamap = ast.literal_eval(question_data[1][2])
            # print question_data[1][3]

            # datamap = eval(question_data[1][3])

            # if isinstance(datamap, dict):
            #     print 'HI!'
            #     print datamap[1]
            # else:
            #     print 'NOPE!', datamap
    ###question_data[id][0]=answer, [id][1]=caegory
    ### [id][2]=question, [id][3]=question(no stopwords)
    ### [id][4]=wikilength

    # Read in training data
    train = DictReader(open("train3.csv", 'r'))
    
    # Split off dev section 80/20
    dev_train = []
    dev_test = []
    position_answers = {}
    final_answers = {}



    for ii in train:
        #Feature dictionary
        d = defaultdict(int)
        # d['Qid'] = ii['question']
        # d['user1'] = int(float(ii['user']))
        d['userpos1'] = abs(int(float(ii['AvgUserPos'])))
        d['qpos1'] = abs(int(float(ii['AvgQuestPos'])))
        d['diff'] = abs(int(float(ii['AvgQuestPos']))) - abs(int(float(ii['AvgUserPos'])))
               
        # d['qpercent'] = myround(float(ii['QuestPercent'])*100, 25)
        # d['upercent'] = myround(float(ii['UserPercent'])*100, 25)

        ##usually not useful
        d['category1'] = question_data[int(ii['question'])][1]
        # d['qans'] = int(float(ii['QuestAnswered']))
        # d['wikilen1'] = int(question_data[int(ii['question'])][4])
        # d['answer_length'] = len(ii['answer'].split(' '))


        #todo: try word/proper nouns before AvgQuest/UserPos

        if int(ii['id']) % 5 == 0:
            ### Positive only
            dev_test.append((d, abs(int(float(ii['position']))), int(ii['id'])))
            final_answers[int(ii['id'])] = int(float(ii['position']))
            ### With negatives
            # dev_test.append((d, int(float(ii['position']))))

        else:
            ### Positive only
            # dev_train.append((d, abs(int(float(ii['position'])))))
            # dev_train.append((d, abs(int(float(ii['RoundPos'])))))
            dev_train.append((d, myround(abs(int(float(ii['position']))), 5)))
            #### With negatives
            # dev_train.append((d, int(float(ii['RoundPos']))))



    # Train a classifier
    print("Training classifier ...")
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)
    # classifier = nltk.classify.MaxentClassifier.train(dev_train, 'GIS', trace=3, max_iter=1)

    predictions = []
    answers = []

    for ii in dev_test:
        # print classifier.classify(ii[0])
        pred = classifier.classify(ii[0])
        predictions.append(pred)
        # predictions.append(ii[0]['qpos1'])
        position_answers[ii[2]] = pred
        answers.append(ii[1])

    rms = mean_squared_error(predictions, answers) ** 0.5
    print "RMS Accuracy Position Only = ", rms

    # # Write predictions
    # o = DictWriter(open('pred.csv', 'w'), ['id', 'pred'])
    # o.writeheader()
    # for ii in sorted(test):
    #     o.writerow({'id': ii, 'pred': test[ii]})

    # classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test)
    # classifier.show_most_informative_features(20)

####################################################
## Correctness Section #############################
####################################################


    
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
    sign_answers = {}


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
                    qids.append(int(ii['id']))


                else:
                    dev_train.append((d, sign(float(ii['position']))))
                    # dev_train.append((d, int(float(ii['RoundPos']))))


        # Train a classifier
        # print("Training classifier ...")
        # classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)
        classifier = nltk.classify.MaxentClassifier.train(dev_train, 'IIS', trace=0, max_iter=20)

        # classifier2.show_most_informative_features(20)

        correct = 0
        total = 0


        for ii in dev_test:
            # print classifier.classify(ii[0])
            ans = classifier.classify(ii[0])
            sign_answers[qids[total]]=sign(ans)
            total+=1
            overall_total+=1
            # question_stats[qids[total-1]][1]+=1
            if sign(ans) == sign(answers[total-1]):
                correct+=1
                overall_correct+=1
                # question_stats[qids[total-1]][0]+=1



    

    predictions = []
    answers = []
    for ii in final_answers:
        answers.append(final_answers[ii])
        # print position_answers[ii]
        # print sign_answers[ii]

        predictions.append(sign_answers[ii]*position_answers[ii])
        # print sign_answers[ii]*position_answers[ii], final_answers[ii]
    
    print "Accuracy = ", float(overall_correct)/overall_total
    rms = mean_squared_error(predictions, answers) ** 0.5
    print "RMS Accuracy = ", rms