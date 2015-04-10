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
    train = DictReader(open("train2.csv", 'r'))
    
    # Split off dev section
    dev_train = []
    dev_test = []
    full_train = []
    wikilen = 0
    qwer = 0
    ee = 0

    for ii in train:
        #Feature dictionary
        d = defaultdict(int)
        # d['Qid'] = ii['question']
        # d['user1'] = int(float(ii['user']))
        d['userpos1'] = int(float(ii['AvgUserPos']))
        d['qpos1'] = int(float(ii['AvgQuestPos']))
        d['diff'] = int(float(ii['AvgQuestPos'])) - int(float(ii['AvgUserPos']))
            
        # print d['wiki_len']
        

        ##usually not useful
        # d['category1'] = question_data[int(ii['question'])][1]
        # d['wikilen1'] = int(question_data[int(ii['question'])][4])

        # for word in question_data[int(ii['question'])][2].split():
        #     word = re.sub(r'[\W_]+', '', word)
        #     word = word.lower()
        #     # print word
        #     d[word] += 1

        ##not so useful 
        # d['length'] = len(question_data[int(ii['question'])][2])
        # d['length'] = len(set(question_data[int(ii['question'])][3]))
        # d['words'] = question_data[int(ii['question'])][3]


        # print question_data[int(ii['question'])][3][0]
        # d['words'] = question_data[int(ii['question'])][3].values()



        #todo: try word/proper nouns before AvgQuest/UserPos

        if int(ii['id']) % 5 == 0:
            ### Positive only
            dev_test.append((d, abs(int(float(ii['position'])))))
            ### With negatives
            # dev_test.append((d, int(float(ii['position']))))

        else:
            ### Positive only
            # dev_train.append((d, abs(int(float(ii['position'])))))
            # dev_train.append((d, abs(int(float(ii['RoundPos'])))))
            dev_train.append((d, myround(abs(int(float(ii['position']))), 5)))
            #### With negatives
            # dev_train.append((d, int(float(ii['RoundPos']))))

        # full_train.append((d, abs(int(float(ii['position'])))))

    # Train a classifier
    print("Training classifier ...")
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)
    # classifier = nltk.classify.MaxentClassifier.train(dev_train, 'GIS', trace=3, max_iter=3)

    total = len(dev_test)
    predictions = []
    answers = []

    for ii in dev_test:
        # print classifier.classify(ii[0])
        predictions.append(classifier.classify(ii[0]))
        answers.append(ii[1])

    rms = mean_squared_error(predictions, answers) ** 0.5
    print "RMS Accuracy = ", rms

    # Retrain on all data
    # classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test)
    classifier.show_most_informative_features(20)