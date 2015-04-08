#!/usr/bin/python
'''
Created on Apr 8, 2015

@author: michael
'''

import argparse
import csv
import nltk
from nltk.corpus import wordnet as wn, brown
import re
from sets import Set
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import mean_squared_error
import sys
import time


def train_taggger():
    print 'starting training'
    start = time.time()
#     print start
    tagger = nltk.tag.TnT()
    tagger.train(brown.tagged_sents())
# #     for ii in brown.tagged_sents():
#     print 'starting tagging'
#     start = time.time()
#     print start
#     tagged_data_words = tagger.tagdata(data)
    end = time.time()
#     print end
    print 'finished training'
    print 'elapsed time: '+str(end-start)+' s'
    return tagger

def tag_question(tagger, question_sentence, question_id):
    print 'starting tagging of question: '+str(question_id)
    start = time.time()
#     print start
    tagged_question = tagger.tag(question_sentence)
    end = time.time()
    print 'finished tagging question: '+str(question_id)
    print 'elapsed time: '+str(end-start) +' s'
    return tagged_question

def prepare_question(question):
#     tokens = question.split()
#     prepared = []
#     for token in tokens:
#         prepared.append(str(token))
#     prepared.append(str(question))
#     print prepared
#     return prepared
    return question.split()


if __name__ == "__main__":
    nltk.download('brown')
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--limit", help="Limit number of questions to preprocess",
                           type=int, default=-1, required=False)
    argparser.add_argument("--in_file", help="Question file name. Default is questions.csv",
                           default='questions.csv', required=False)
    argparser.add_argument("--out_file", help="Output file name. Default is questions_processed.csv",
                           default='questions_processed.csv', required=False)    
    argparser.add_argument("--start_position", help="Row in question csv file to start at",
                          type=int, default=0, required=False)
    argparser.add_argument("--end_position", help="Row in question csv file to end at",
                          type=int, default=-1, required=False)
    args = argparser.parse_args()
    
    in_file_name = args.in_file
    infile = open(in_file_name)
    reader = csv.DictReader(infile, ['id', 'answer', 'type', 'category', 'question', 'keywords'])
    
    out_file_name = args.out_file
    outfile = open(out_file_name, 'w')
    writer = csv.DictWriter(outfile, ['id', 'answer', 'type', 'category', 'question', 'keywords'])
    
    writer.writeheader()
    tagger = train_taggger()
    
    count = 0;
    limit = args.limit
    start = args.start_position
    end = args.end_position
    position_count = 0;
    for row in reader:
        if position_count < start:
            position_count = position_count + 1
            continue
        if (limit > 0 and count >= limit) or (end > 0 and position_count >= end):
            break
        id = row['id']
        answer = row['answer']
        type = row['type']
        category = row['category']
        question = row['question']
        print question
        keywords = row['keywords']
        prepared_question = prepare_question(question)
        tagged_question = tag_question(tagger, prepared_question, id)
        print tagged_question
        writer.writerow({'id':id, 'answer':answer,'type':type,'category':category,'question':tagged_question, 'keywords':keywords})
        count = count + 1
        position_count = position_count + 1
        
    infile.close()
    outfile.close()
    print 'Finished'
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    