#!/usr/bin/python
import argparse
import nltk
from Predictor import Predictor

if __name__ == "__main__":
    nltk.download('wordnet')
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--local", help="Use training data as a test of the model",
                           action='store_true', default=True, required=False)
    argparser.add_argument("--local_selection", help="1 in <selection number> to select from training data to use as test data",
                           type=int, default=10, required=False)
#     argparser.add_argument("--default_vectorizer", help="Use the default vectorizer provided with the assignment",
#                            action='store_true', default=False, required=False)
    argparser.add_argument("--limit", help="Limit the number of values used in local mode",
                           type=int, default=-1, required=False)
    argparser.add_argument("--predict", help="Predict on the Kaggle test data", 
                           action='store_true', default=False, required=False)
    argparser.add_argument("--wiki_file", help="File containing stored Wikipedia data", 
                           action='store_true', default="wiki_data.pkl", required=False)
    argparser.add_argument("--regenerate_wiki", help="Regenerate the Wikipedia data", 
                           action='store_true', default=False, required=False)
    args = argparser.parse_args()
    
    features_abs = {'word':False, 'speech':True, 'capital':True, 'all_upper':False, 'foreign':True, 
                'unique':True, 'ngram_range':(2,20), 'user_average':True,
                'numbers':False, 'before_noun':True, 'wiki_answer':True, 'question_count':False, 'question_average':True,
                'question_percent':False, 'provided_answer':True, 'category_average':True, 'question_answer_percent':True,
                'user_category_average':False, 'question_length':True}
    
    features_sign = {'word':False, 'speech':True, 'capital':True, 'all_upper':False, 'foreign':True, 
                'unique':True, 'ngram_range':(2,20), 'user_average':True,
                'numbers':False, 'before_noun':True, 'wiki_answer':True, 'question_count':False, 'question_average':True,
                'question_percent':False, 'provided_answer':True, 'category_average':True, 'question_answer_percent':True,
                'user_category_average':True, 'kernel':'sigmoid', 'question_length':True}
    
    predictor = Predictor(args, features_abs, features_sign, cluster=False)
    predictor.run()


