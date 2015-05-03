#!/usr/bin/python
import argparse
from csv import DictReader
from distlib.util import CSVReader
import nltk

from Predictor import Predictor


if __name__ == "__main__":
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
    argparser.add_argument("--load_best_features", help="Load the best features for each cluster from a records file",
                           action='store_true', default=False, required=False)
    argparser.add_argument("--records_file", help="Records file to load best features from",
                           action='store_true', default="records.csv", required=False)
    args = argparser.parse_args()
    nltk.download('wordnet')
    
    cluster_boundaries = [10,50,100,150,200,500, 1000, 5000]
    if args.load_best_features:
        records_f = open(args.records_file, 'r')
        reader = DictReader(records_f)
        features_abs = []
        features_sign = []
        min_errors = {}
        min_errors_features = {}
        correct_ranges = []
        for row in reader:
            ranges = row['Range Errors']
            features = row['Features']
            features_tokens = features.split(" Binary:")
            continuous_features = features_tokens[0].replace("Continuous:", "").replace("Continuous", "")
            binary_features = features_tokens[1]
            continuous_features_list = eval(continuous_features)
            binary_features_list = eval(binary_features)
            ranges_tokens = ranges.split("|")
            if len(ranges_tokens) != len(cluster_boundaries):
                continue
            else:
                correct_ranges = ranges
            for range_index in range(len(ranges_tokens)):
                current_error = float(ranges_tokens[range_index].split(":")[1])
#                 print "Current error: "+str(current_error)
                if range_index in min_errors and min_errors[range_index] > current_error or range_index not in min_errors:
                        min_errors[range_index] = current_error
                        min_errors_features[range_index] = [continuous_features_list[range_index], binary_features_list[range_index]]
        for key in sorted(min_errors):
            print "Error index: "+str(key)+" Error: "+str(min_errors[key])+" Range: "+str(correct_ranges.split("|")[key].split(":")[0])
            features_abs.append(min_errors_features[key][0])
            features_sign.append(min_errors_features[key][1])
        records_f.close()
    else:
        
        features_abs_10 = {'word':False, 'speech':True, 'capital':True, 'all_upper':True, 'foreign':True, 
                    'unique':True, 'ngram_range':(1,1), 'user_average':True, 'wiki_num_results':True,
                    'numbers':False, 'before_noun':True, 'wiki_answer':True, 'question_count':False, 'question_average':True,
                    'question_percent':False, 'provided_answer':False, 'category_average':True, 'question_answer_percent':True,
                    'user_category_average':True, 'question_length':True, 'question_mark':False, 'question_sentence_count':False,
                    'question_comma_count':False, 'user_num_answered':False, 'question_double_quote_count':True, 
                    'question_single_quote_count':False, 'question_asterisk_count':False,
                    'user_num_incorrect':False, 'user_incorrect_average':True, 'user_correct_average':False, 
                    'syllable_count':False, 'grade_level':False, 'period_count':False, 'question_incorrect_average':True,
                    'question_correct_average':False, 'perform_binary_classification':True,'absolute_continuous_y':True, 
                    'question_length_average_difference':True, 'cap_question_absolute_average':True, 'invert_cap_sign':False,
                    'cap_question_absolute_average':True}
        
        features_abs_50 = {'word':False, 'speech':False, 'capital':True, 'all_upper':True, 'foreign':True, 
                    'unique':True, 'ngram_range':(1,1), 'user_average':True, 'wiki_num_results':True,
                    'numbers':False, 'before_noun':True, 'wiki_answer':False, 'question_count':False, 'question_average':True,
                    'question_percent':False, 'provided_answer':False, 'category_average':True, 'question_answer_percent':True,
                    'user_category_average':True, 'question_length':True, 'question_mark':False, 'question_sentence_count':True,
                    'question_comma_count':True, 'user_num_answered':True, 'question_double_quote_count':True, 
                    'question_single_quote_count':False, 'question_asterisk_count':False,
                    'user_num_incorrect':True, 'user_incorrect_average':False, 'user_correct_average':False,
                    'syllable_count':False, 'grade_level':True, 'period_count':False, 'perform_binary_classification':True,'absolute_continuous_y':False,
                    'question_length_average_difference':True, 'cap_question_absolute_average':True, 'invert_cap_sign':False,
                    'cap_question_absolute_average':True}
        
        features_abs_100 = {'word':False, 'speech':False, 'capital':True, 'all_upper':True, 'foreign':True, 
                    'unique':True, 'ngram_range':(2,20), 'user_average':True, 'wiki_num_results':True,
                    'numbers':False, 'before_noun':True, 'wiki_answer':False, 'question_count':False, 'question_average':False,
                    'question_percent':True, 'provided_answer':False, 'category_average':True, 'question_answer_percent':True,
                    'user_category_average':False, 'question_length':True, 'question_mark':False, 'question_sentence_count':True,
                    'question_comma_count':True, 'user_num_answered':True, 'question_double_quote_count':True, 
                    'question_single_quote_count':False, 'question_asterisk_count':False,
                    'user_num_incorrect':True, 'user_incorrect_average':False, 'user_correct_average':False, 
                    'syllable_count':False, 'grade_level':True, 'period_count':False, 'perform_binary_classification':True,'absolute_continuous_y':False,
                    'question_length_average_difference':True, 'cap_question_absolute_average':True, 'invert_cap_sign':False,
                    'cap_question_absolute_average':True}
        
        features_abs_200 = {'word':False, 'speech':False, 'capital':True, 'all_upper':True, 'foreign':True, 
                    'unique':True, 'ngram_range':(1,1), 'user_average':True, 'wiki_num_results':True,
                    'numbers':False, 'before_noun':True, 'wiki_answer':True, 'question_count':True, 'question_average':True,
                    'question_percent':False, 'provided_answer':False, 'category_average':True, 'question_answer_percent':True,
                    'user_category_average':True, 'question_length':True, 'question_mark':False, 'question_sentence_count':False,
                    'question_comma_count':False, 'user_num_answered':True, 'question_double_quote_count':True, 
                    'question_single_quote_count':False, 'question_asterisk_count':False,
                    'user_num_incorrect':True, 'user_incorrect_average':True, 'user_correct_average':False, 
                    'syllable_count':False, 'grade_level':True, 'period_count':False, 'perform_binary_classification':True,'absolute_continuous_y':False,
                    'question_length_average_difference':True, 'cap_question_absolute_average':True, 'invert_cap_sign':False,
                    'cap_question_absolute_average':True}
        
        features_abs_1000 = {'word':False, 'speech':False, 'capital':True, 'all_upper':True, 'foreign':True, 
                    'unique':True, 'ngram_range':(1,1), 'user_average':True, 'wiki_num_results':True,
                    'numbers':False, 'before_noun':True, 'wiki_answer':True, 'question_count':False, 'question_average':True,
                    'question_percent':False, 'provided_answer':False, 'category_average':True, 'question_answer_percent':True,
                    'user_category_average':False, 'question_length':True, 'question_mark':False, 'question_sentence_count':True,
                    'question_comma_count':True, 'user_num_answered':True, 'question_double_quote_count':True, 
                    'question_single_quote_count':False, 'question_asterisk_count':False,
                    'user_num_incorrect':True, 'user_incorrect_average':False, 'user_correct_average':False, 
                    'syllable_count':False, 'grade_level':True, 'period_count':False, 'perform_binary_classification':True,'absolute_continuous_y':False,
                    'question_length_average_difference':True, 'cap_question_absolute_average':True, 'invert_cap_sign':False,
                    'cap_question_absolute_average':True}
        features_abs_150 = features_abs_100
        features_abs_500 = features_abs_100
        features_abs_5000 = features_abs_100
        
        features_abs = [features_abs_10, features_abs_50, features_abs_100, features_abs_150, features_abs_200,
                         features_abs_500, features_abs_1000, features_abs_5000, features_abs_100]
        
    
#     features_sign_10 = {'word':False, 'speech':False, 'capital':True, 'all_upper':True, 'foreign':True, 
#                     'unique':True, 'ngram_range':(2,20), 'user_average':True, 'wiki_num_results':True,
#                     'numbers':True, 'before_noun':True, 'wiki_answer':True, 'question_count':True, 'question_average':True,
#                     'question_percent':True, 'provided_answer':False, 'category_average':True, 'question_answer_percent':True,
#                     'user_category_average':True, 'question_length':True, 'question_mark':False, 'question_sentence_count':False,
#                     'question_comma_count':False, 'user_num_answered':True, 'question_double_quote_count':False, 
#                     'question_single_quote_count':False, 'question_asterisk_count':False, 'previous_prediction':True,
#                     'user_num_incorrect':True, 'user_incorrect_average':True, 'user_correct_average':True, 
#                     'syllable_count':True, 'grade_level':True, 'period_count':True}
        features_sign_10 = {'all_upper': True, 'user_num_incorrect': True, 'question_single_quote_count': False, 'user_category_average': True, 'period_count': False, 
                        'question_answer_percent': True, 'ngram_range': (2, 10), 'user_incorrect_average': True, 'numbers': True, 'user_num_answered': True, 
                        'question_asterisk_count': False, 'question_length': True, 'unique': True, 'user_average': True, 'question_count': True, 'before_noun': True, 
                        'grade_level': False, 'question_sentence_count': False, 'word': False, 'provided_answer': True, 'category_average': True, 'question_mark': False, 
                        'user_correct_average': True, 'wiki_answer': True, 'wiki_num_results':True, 'previous_prediction': True, 'foreign': True, 'question_percent': True, 'speech': False, 
                        'capital': True, 'question_comma_count': True, 'question_double_quote_count': True, 'question_average': True, 
                        'syllable_count': False, 'question_length_average_difference':True, 'binary_classifier':'svm',}
        features_sign_200 = {'all_upper': True, 'user_num_incorrect': True, 'question_single_quote_count': False, 'user_category_average': True, 'period_count': False, 
                        'question_answer_percent': True, 'ngram_range': (1,1), 'user_incorrect_average': True, 'numbers': True, 'user_num_answered': True, 
                        'question_asterisk_count': False, 'question_length': True, 'unique': True, 'user_average': True, 'question_count': True, 'before_noun': True, 
                        'grade_level': False, 'question_sentence_count': False, 'word': False, 'provided_answer': True, 'category_average': True, 'question_mark': False, 
                        'user_correct_average': True, 'wiki_answer': True, 'wiki_num_results':True, 'previous_prediction': True, 'foreign': True, 'question_percent': True, 'speech': False, 
                        'capital': True, 'question_comma_count': True, 'question_double_quote_count': True, 'question_average': True, 
                        'syllable_count': False, 'binary_classifier':'svm', 'question_length_average_difference':True}
        features_sign = [features_sign_10, features_sign_10, features_sign_10, features_sign_10, features_sign_200,
                      features_sign_10, features_sign_10, features_sign_200, features_sign_200]
    
    predictor = Predictor(args, features_abs, features_sign, cluster=True, n_estimators=500,
                          cluster_boundaries=cluster_boundaries)#, skip_clusters=[1,2,3,4,5,6,7,8])
    predictor.run()


