#!/usr/bin/python
'''
Created on Apr 9, 2015

@author: michael
'''
from _ast import Dict
import argparse
import csv
import nltk
import preprocess_data


if __name__ == "__main__":
    #7949 lines in questions.csv -> 7948 questions
    #-> 1987 questions for 4 threads
    nltk.download('brown')
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--in_files", help="Comma-separted list of files to be combined",
                           default="", required=False)
    argparser.add_argument("--question_file", help="FIle containing question data. Default is questions.csv",
                           default="questions.csv", required=False)
    argparser.add_argument("--out_file", help="Output file name. Default is questions_processed.csv",
                           default='questions_processed.csv', required=False)    
    args = argparser.parse_args()
    
    files = args.in_files.split(",")
    if len(files) < 1:
        print "No input files received. Exiting."    
    processed_data = Dict()
    missing_data = Dict()
    
    for in_file in files:
        infile = open(in_file)
        reader = csv.DictReader(infile, ['id', 'answer', 'type', 'category', 'question', 'keywords'])
        for row in reader:
            processed_data[int(row['id'])] = row
        infile.close()
    
    infile = open(args.question_file)
    reader = csv.DictReader(infile, ['id', 'answer', 'type', 'category', 'question', 'keywords'])
    for row in reader:
        if int(row['id']) not in processed_data:
            missing_data[int(row['id'])] = row
            
    tagger = preprocess_data.train_taggger()
    for question_data in missing_data:
        raw_question =  question_data['question']
        prepared_question = preprocess_data.prepared_question(raw_question)
        tagged_question = preprocess_data.tag_question(tagger, prepared_question, question_data['id'])
        processed_data[int(question_data['id'])] = {'id':question_data['id'], 'answer':question_data['answer'],
                                                    'type':question_data['type'],'category':question_data['category'],
                                                    'question':tagged_question, 'keywords':question_data['keywords']}
    
    out_file_name = args.out_file
    outfile = open(out_file_name, 'w')
    writer = csv.DictWriter(outfile, ['id', 'answer', 'type', 'category', 'question', 'keywords'])
    writer.writeheader()
    for question_data in processed_data:
        writer.writerow(question_data)
    
    print "Finished"
        