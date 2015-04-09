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
import sys


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
    
    if len(args.in_files) < 1:
        print "No input files received. Exiting."
        sys.exit()
    files = args.in_files.split(",")
    processed_data = {}
    missing_data = {}
    
    for in_file in files:
        infile = open(in_file)
        reader = csv.DictReader(infile)
        for row in reader:
        #    print row['id']
            processed_data[int(row['id'])] = row
        infile.close()
    
    infile = open(args.question_file)
    reader = csv.DictReader(infile,['id', 'answer', 'type', 'category', 'question', 'keywords'])
    for row in reader:
        if int(row['id']) not in processed_data:
            missing_data[int(row['id'])] = row
    infile.close()
         
    print "Number of questions missed: "+ str(len(missing_data))

    tagger = preprocess_data.train_taggger()
    for question_id in missing_data:
        raw_question =  missing_data[question_id]['question']
        prepared_question = preprocess_data.prepare_question(raw_question)
        tagged_question = preprocess_data.tag_question(tagger, prepared_question, question_id)
        processed_data[question_id] = {'id':missing_data[question_id]['id'], 'answer':missing_data[question_id]['answer'],
                                                    'type':missing_data[question_id]['type'],'category':missing_data[question_id]['category'],
                                                    'question':tagged_question, 'keywords':missing_data[question_id]['keywords']}
    
    out_file_name = args.out_file
    outfile = open(out_file_name, 'w')
    writer = csv.DictWriter(outfile, ['id', 'answer', 'type', 'category', 'question', 'keywords'])
    writer.writeheader()
    for question_id in processed_data:
        writer.writerow(processed_data[question_id])
    outfile.close()

    print "Finished"
        
