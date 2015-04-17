
from _collections import defaultdict
import argparse
from collections import Counter
from csv import DictWriter
import csv
import datetime
from distlib.util import CSVWriter
from math import ceil
import nltk
from nltk.corpus import wordnet as wn
from numpy import sign
import numpy
import os
import pickle
import re
from scipy.sparse.csgraph._min_spanning_tree import csr_matrix
from sets import Set
from sklearn import linear_model, svm, neural_network, ensemble
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import mean_squared_error
import sys
import time
import wikipedia

import numpy as np


VALID_POS = ['FW', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 
             'NNP', 'NNPS', 'PDT', 'POS', 'RB', 'RBR', 'RBS',
             'RP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WRB']

class Analyzer:
    def __init__(self, features):
        self.features = defaultdict(lambda: False)
        for feature in features:
            self.features[feature] = features[feature]
        print self.features
        self.numeric_features = defaultdict(dict)
        self.index = 0
    
    def strip_tags_from_words(self, word_list):
        stripped_words = []
        for word_feature in word_list:
            stripped_words.append(word_feature.split(":")[0])
        return stripped_words
    
    def store_numeric_feature(self, example_index, feat_name, numeric_feature):
        self.numeric_features[example_index][feat_name] = float(numeric_feature)
        
    def process_question_string(self, feat_id, question_string):
        question_features = question_string.split()
        if self.features['ngram_range'] != (1,1):
            current_words = self.strip_tags_from_words(question_features)
            for ngram_length in range(self.features['ngram_range'][0], self.features['ngram_range'][1]+1):
                    for ngram in zip(*[current_words[i:] for i in range(3)]):
                        ngram = " ".join(ngram)
                        yield ngram
        
        for feat_tuple in question_features:
            feat_tokens = feat_tuple.split(":")
            if self.features['word']: #and self.ngram_range == (1,1):
                yield feat_tokens[0]
                
            if self.features['speech']:
                if feat_tokens[1] != "Unk":
                    yield feat_tokens[1]
                    
            if self.features['capital']:
                if feat_tokens[2] == "C":
                    yield feat_tokens[2]
                    
            if self.features['all_upper']:
                if feat_tokens[3] == "U":
                    yield feat_tokens[3]
                    
            if self.features['foreign']:
                if feat_tokens[4] == "F":
                    yield feat_tokens[4]
                    
            if self.features['unique']:
                if feat_tokens[5] == "UN":
                    yield feat_tokens[5]
                    
            if self.features['numbers']:
                if feat_tokens[6] == "NUM":
                    yield feat_tokens[6]
                    
            if self.features['before_noun']:
                if feat_tokens[7] == "BFRN":
                    yield feat_tokens[7]
                    
            if self.features['question_count']:
                self.store_numeric_feature(feat_id, 'question_count', int(feat_tokens[8]))
            
            if self.features['question_average']:
                self.store_numeric_feature(feat_id, 'question_average', float(feat_tokens[9]))
                
            if self.features['question_percent']:
                self.store_numeric_feature(feat_id, 'question_percent', float(feat_tokens[10]))
                    
            if self.features['use_dictionary']:
                for category in self.dictionary:
                    if feat_tokens[0] in self.dictionary[category]:
                        yield category.upper()
                        
    def process_user_string(self,feat_id, user_string):
        user_features_tokens = user_string.split(":")
        if self.features['user_average']:
            if user_features_tokens[0] == "USR_AVG":
                self.store_numeric_feature(feat_id, 'user_average', float(user_features_tokens[1]))

    def process_answer_string(self, feat_id, answer_string):
        answer_features_tokens = answer_string.split(":")
        if self.features['wiki_answer']:
            if answer_features_tokens[2] == "WIKI_FIRST":
                self.store_numeric_feature(feat_id, 'wiki_first', int(answer_features_tokens[3]))
        if self.features['provided_answer']:
            self.store_numeric_feature(feat_id, 'provided_answer', int(answer_features_tokens[4]))

    def process_category_string(self, feat_id, category_string):
        category_features_tokens = category_string.split(":")
        if self.features['category_average']:
            self.store_numeric_feature(feat_id, 'category_average', float(category_features_tokens[0]))
    
    def add_numeric_features(self, feature_matrix):
        feature_array = feature_matrix.toarray()
        new_feature_array = []
        for i in range(len(feature_array)):
            current_features = feature_array[i].tolist()
            for feat_index in self.numeric_features[i]:
                current_features.append(self.numeric_features[i][feat_index])
            new_feature_array.append(current_features)
        return csr_matrix(new_feature_array)
    
    def __call__(self, feature_string):
        feature_strings_ = feature_string.split("#")
        feat_id = int(feature_strings_[0])
        feature_strings = feature_strings_[1].split("|")
        
        for question_feat in self.process_question_string(feat_id, feature_strings[1]):
            yield question_feat
        
        self.process_user_string(feat_id, feature_strings[0])
#         for user_feat in self.process_user_string(feat_id, feature_strings[0]):
#             yield user_feat
            
        self.process_answer_string(feat_id, feature_strings[2])    
#         for answer_feat in self.process_answer_string(feat_id, feature_strings[2]):
#             yield answer_feat
        self.process_category_string(feat_id, feature_strings[3])
#         for category_feat in self.process_category_string(feat_id, feature_strings[3]):
#             yield category_feat
            
            
        

class Example:
    def __init__(self):
        self.id = 0
        self.question = ""
        self.user = 0
        self.answer = ""
        #observed value of response time from training data
        self.observation =  0
        #predicted value obtained from classifier
        self.prediction = 0

class Category:
    def __init__(self, category):
        self.category = category
        self.questions = []
        self.average = 0.0
        self.abs_total = 0.0
        self.total = 0.0
        self.count = 0
        self.users = []
        
    def add_question(self, q_id):
        self.questions.append(q_id)
        
    def add_occurrence(self, user_container, user_response):
        self.count += 1
        self.total += user_response
        self.abs_total += abs(user_response)
        self.average = float(self.total)/float(self.count)
        self.users.append(user_container)

        
class Question:
    def __init__(self, question, category, keywords, answer, q_type, q_id):
        self.question = question
        self.category = category
        self.keywords = keywords
        self.answer = answer
        self.q_type = q_type
        self.q_id = int(q_id)
        self.num_correct = 0
        self.num_incorrect = 0
        self.average_response = 0.0
        self.absolute_average = 0.0
        self.percent_correct = 0
        self.count = 0
        self.running_total = 0.0
        self.running_magnitude_total = 0.0
        
    def add_question_occurence(self, response_time):
        if response_time == 0:
            response_time = self.absolute_average
#         current_total = float(self.average_response) * float(self.count)
        self.running_total += response_time
        self.running_magnitude_total += abs(response_time)
        self.count += 1
        if response_time > 0:
            self.num_correct += 1
        else:
            self.num_incorrect += 1
            
        self.percent_correct = float(self.num_correct) / float(self.count)
        self.absolute_average = (float(self.running_total) + (float(response_time)))/float(self.count)
        self.average_response = (float(self.running_magnitude_total) + abs(float(response_time)))/float(self.count)
 
 
        
class User:
    def __init__(self, u_id):
        self.u_id = u_id
        self.average_position = 0
        self.num_questions = 0
        self.num_correct = 0
        self.num_incorrect = 0
        self.questions = {}
        
    def add_question(self, q_id, correct, position, answer):
        self.questions[q_id] = {'q_id':q_id, 'correct':correct, 'position':position, 'answer':answer}
        current_total = self.num_questions#self.num_correct + self.num_incorrect
        current_position_sum = self.average_position*current_total
        updated_total = current_total + 1
        updated_sum = current_position_sum + position
        self.average_position = updated_sum/updated_total
        if correct:
            self.num_correct = self.num_correct + 1
        else:
            self.num_incorrect = self.num_incorrect + 1
        self.num_questions = self.num_questions + 1


        
class Featurizer:
    def __init__(self, category_dict, features=defaultdict(lambda: False), use_default=False, analyzer=u'word'):
        self.stop_words=['uner', 'via', 'answer', 'ANSWER', 'was', 'his','the','that','and', 'points', 'pointsname', 'this', 'for', 'who', 'they','can', 'its', 'also', 'these', 'are', 'name', 'after', 'than', 'had', 'with', 'about', 'ftp', '10', '15', 'this', 'from', 'not', 'has', 'well']
        self.analyzer = analyzer
        self.category_dict = category_dict
        self.features = defaultdict(lambda: False)
        for feat in features:
            self.features[feat] = features[feat]
        
        if use_default:
            self.vectorizer = CountVectorizer(ngram_range=(1,10), stop_words=self.stop_words)
            self.default = True
        else:
            self.vectorizer = CountVectorizer(analyzer=analyzer)
            self.default = False


    def train_feature(self, examples, users, wiki_data, limit=-1):
        count_features = self.vectorizer.fit_transform(ex for ex in all_examples(limit, examples, users, self.features, self.category_dict, wiki_data, default = self.default))
        if not self.default:
            return self.analyzer.add_numeric_features(count_features)
        return count_features
            

    def test_feature(self, examples, users, wiki_data, limit=-1):
        count_features = self.vectorizer.transform(ex for ex in all_examples(limit, examples, users, self.features, self.category_dict, wiki_data, default = self.default))
        if not self.default:
                return self.analyzer.add_numeric_features(count_features)
        return count_features

    def show_topN(self, classifier, categories, N=10):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            if(len(classifier.coef_) - 1 >= i):
                current = np.argsort(classifier.coef_[i])
                if len(current) > N:
                    topN = current[-N:]
                else:
                    topN = current
                print("%s: %s" % (category, " ".join(feature_names[topN])))
            else:
                print 'Could not classify ' + str(category)
                
    def show_top10(self, classifier, categories):
        self.show_topN(classifier, categories, 10)
            
    def show_top20(self, classifier, categories):
        self.show_topN(classifier, categories, 20)



def all_examples(limit, examples, users, features, category_dict, answer_wiki_features, default=False):
    for i in range(len(examples)):
        current_example = examples[i]
        current_user = users[current_example.user]
        tagged_example_str = get_example_features(i,features, category_dict, current_example.question, current_user, answer_wiki_features, default) 
        yield tagged_example_str


        
def get_user_features(features, user):
    
    average = str(abs(user.average_position))
    user_feature = ""
    if features['user_average']:
        user_feature += "USR_AVG:"+average 
    return user_feature

def get_category_features(pred_features, category, category_dict):
    category_string = ""
    if pred_features['category_average']:
        category_string += str(category_dict[category].average)
    return category_string

        
def get_question_features(features, question_container):
    question = question_container.question
    category = question_container.category
    keywords = question_container.keywords
    answer = question_container.answer
    question_id = question_container.q_id
    
    annotated_words = []
    stop_words=['uner', 'via', 'answer', 'ANSWER', 'was', 'his','the','that','and', 'points', 'pointsname', 'this', 'for', 'who', 'they','can', 'its', 'also', 'these', 'are', 'name', 'after', 'than', 'had', 'with', 'about', 'ftp', '10', '15', 'this', 'from', 'not', 'has', 'well']
    prev_word = ""
    prev_word_unannotated = ""
    reg_exp = re.compile(u'(?u)\\b[a-zA-Z]{2}[a-zA-Z]+|[0-9]{4}\\b')
    reg_exp_alphanum = re.compile(u'\\w')

    words = {}
    pos = {}
    
    word_index = 0
    for word_tuple in question:
        words[word_index] = word_tuple[0]
        pos[word_index] = word_tuple[1]
        word_index = word_index + 1
    
    c = Counter([values for values in words.itervalues()])
    counts = dict(c)
    
    encounteredNoun = False
        
    for word_index in words:
        word = words[word_index]
        result = reg_exp.match(word)
        if result is None:
            continue
        original_annotated_word = result.string[result.start(0):result.end(0)]
        annotated_word = ""
        if features['word']:
            annotated_word = original_annotated_word
        if original_annotated_word.lower() in stop_words:
            prev_word_unannotated = word
            continue
        annotated_word += ":"
        if features['speech'] and word_index in pos:
            if pos[word_index] in VALID_POS:
                annotated_word += pos[word_index]
                if pos[word_index] in ['NN', 'NNS', 'NNP', 'NNPS']:
                    encounteredNoun = True
        annotated_word += ":"
        if features['capital'] and original_annotated_word[0].isupper() and len(prev_word_unannotated) > 0 and reg_exp_alphanum.match(prev_word_unannotated[len(prev_word_unannotated)-1]):
            annotated_word+="C"
        annotated_word += ":"
        if features['all_upper'] and original_annotated_word.isupper():
            annotated_word+="U"
        annotated_word += ":"
        if features['foreign'] and not wn.synsets(original_annotated_word):
            annotated_word+="F"
        annotated_word += ":"
        

        if features['unique'] and word in counts:
            if counts[word] == 1:
                annotated_word+="UN"
        annotated_word += ":"        
        #count number of words that are numberic
        if features['numbers']  and original_annotated_word.isdigit():
            annotated_word+="NUM"
        annotated_word += ":"
        if features['before_noun'] and not encounteredNoun:
            annotated_word+="BFRN"
        annotated_word += ":"
        if features['question_count']:
            annotated_word += str(int(question_container.count))
        annotated_word += ":"
        if features['question_average']:
            annotated_word += str(question_container.average_response)
        annotated_word += ":"
        if features['question_percent']:
            annotated_word += str(question_container.absolute_average)
        
        
        annotated_words.append(annotated_word)
        prev_word = original_annotated_word
        prev_word_unannotated = word
        
    tagged = " ".join(annotated_words)
    return tagged

def get_answer_features(features, answer, answer_wiki_features, provided_answer):
    annotated_words = ""
    question_string = "'"
    incorrect_threshold = 10
    if features['wiki_answer']:
        if answer in answer_wiki_features:
            num_results = answer_wiki_features[answer][0]
            first_len = answer_wiki_features[answer][1]
            num_occurences = answer_wiki_features[answer][2]
            question_string += "WIKI_NUM:"+str(num_results)+":WIKI_FIRST:"+str(first_len)+":"+str(num_occurences)
        else:
            try:
                results, suggestion = eval(str(wikipedia.search(answer, results=1000000, suggestion=True)))
                if suggestion is not None:
                    print "Suggestion: "+str(suggestion)+" "+str(len(results))+" answer: "+answer
                    if len(results) <= incorrect_threshold:
                        results = eval(str(wikipedia.search(str(suggestion), results=1000000)))
                num_results = len(results)
                question_string += "WIKI_NUM:"+str(num_results)+":WIKI_FIRST:"
                if num_results > 0:
                    first_result = wikipedia.page(str(results[0]))
                    first_len = len(first_result.content)
                    question_string+=str(first_len)
                    
                    if features['provided_answer'] and len(provided_answer) > 0:
                        num_occurences = results.lower().count(provided_answer.lower())
                        print num_occurences
                        question_string += ":"+str(num_occurences)
                    else:
                        num_occurences = 0
                        question_string += ":0"
                    answer_wiki_features[answer] = [num_results,first_len, num_occurences, results, first_result]
                else:
                    question_string+="0:0"
                    answer_wiki_features[answer] = [num_results, 0, 0, results, None]
            except:
                question_string = "WIKI_NUM:0:WIKI_FIRST:0:0"
                answer_wiki_features[answer] = [0,0, 0, None, None]
    else:
        question_string = "::::0"
    annotated_words += question_string+":"
    return annotated_words


def get_example_features(example_id, features, category_dict, question, user, answer_wiki_features, default=False):
    
    question_features = get_question_features(features, question)
    user_features = get_user_features(features, user)
    answer = question.answer
    question_id = question.q_id
    user_questions = user.questions
    if question_id in user_questions:
        provided_answer = user.questions[question.q_id]['answer']
    else:
        provided_answer = ""
    answer_features = get_answer_features(features, answer, answer_wiki_features, provided_answer)
    current_category = question.category
    category_features = get_category_features(features, current_category, category_dict)
    if default:
        feature_list = []
        question_feature_list = question_features.split()
        for feature in question_feature_list:
            individual_features = feature.split(":")
            for current_individual_feature in individual_features:
                feature_list.append(current_individual_feature)
        user_features_list = user_features.split()
        for feature in user_features_list:
            individual_features = feature.split(":")
            for current_individual_feature in individual_features:
                feature_list.append(current_individual_feature)
#         answer_features_list = answer_features
#         for feature in answer_
        feature_string = " ".join(feature_list)
    else:
        feature_string = str(example_id)+"#"+user_features +"|"+question_features+"|"+answer_features+"|"+category_features
    return feature_string


def stringToInt(s):
    return int(float(s))


def rootMeanSquaredError(examples):
    predictions = [x.prediction for x in examples]
    observations = [x.observation for x in examples]
    return mean_squared_error(predictions, observations) ** 0.5


def producePredictions(trainingExamples, testExamples, users, continuous_features, binary_features, wiki_data, category_dict, average):
    if 'kernel' in features_sign:
        kernel = features_sign['kernel']
    else:
        kernel = 'rbf'
        features_sign['kernel'] = 'rbf'
        
    if 'gamma' in features_sign:
        gamma = float(features_sign['gamma'])
    else:
        gamma = 0.0
        features_sign['gamma'] = str(gamma)
    analyzer_abs = Analyzer(features=continuous_features)
    featurizer_abs = Featurizer(category_dict, features=continuous_features, use_default=False, analyzer=analyzer_abs)
    
    analyzer_sign = Analyzer(features=binary_features)
    featurizer_sign = Featurizer(category_dict, features=binary_features, use_default=False, analyzer=analyzer_sign)
#     featurizer = Featurizer(use_default=True)
    y_train_abs = []
    y_train_sign = []
    for train_example in trainingExamples:
        y_train_abs.append(train_example.observation)
#         y_train_abs.append(abs(train_example.observation))
#         y_train_sign.append(sign(train_example.observation))
    print "Generating continuous training x"
    x_train_abs = featurizer_abs.train_feature(trainingExamples, users, wiki_data)
    print featurizer_abs.vectorizer.vocabulary_.get('document')
#     for feat in x_train:
#         print feat
    print x_train_abs.toarray()[0]
    print x_train_abs.toarray()[50]
# #     print x_train.toarray()[4000]
# #     del trainingExamples
    print "Generating continuous test x"
    x_test_abs = featurizer_abs.test_feature(testExamples, users, wiki_data)
      
      
#     classifier = SGDClassifier(loss='log', penalty='l2', shuffle=True)
#     
#     lr.fit(x_train, y_train)
  
#     continuous_classifier = linear_model.LinearRegression()
#     continuous_classifier = svm.NuSVR()
    continuous_classifier = ensemble.GradientBoostingRegressor(n_estimators=650)
#     continuous_classifier = ensemble.AdaBoostRegressor()
#     continuous_classifier = ensemble.RandomForestRegressor()
#     continuous_classifier = ensemble.BaggingRegressor()
      
    print "Fitting continuous classifier"  
    continuous_classifier.fit(x_train_abs.toarray(), y_train_abs)  
      
    print "Fit continuous training data"
#     
    print "Predicting continuous classifier"
    predictions_abs = continuous_classifier.predict(x_test_abs.toarray())
#     print predictions_abs
    
#     binary_classifier = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    
#     binary_classifier = svm.SVC(kernel=kernel, class_weight='auto')
#     binary_classifier = svm.SVC(kernel=kernel)
# #     binary_classifier = neural_network.BernoulliRBM()
#     
#     print "Generating binary training x"
#     x_train_sign = featurizer_sign.train_feature(trainingExamples, users, wiki_data)
#     print x_train_sign.toarray()[0]
#     
#     print "Generating binary text x"
#     x_test_sign = featurizer_sign.test_feature(testExamples, users, wiki_data)
#     print x_test_sign.toarray()[0]
#     print "Fitting binary classifier"
#     binary_classifier.fit(x_train_sign, y_train_sign)
#     
#     print "Fit binary classifier"
#     print "Predicting binary classifier"
#     predictions_binary = binary_classifier.predict(x_test_sign)
    
#     print predictions_binary
    
    for i in range(len(predictions_abs)):
        testExamples[i].prediction = int(ceil(predictions_abs[i]))
    
#     for i in range(len(predictions_binary)):
#         current_sign = predictions_binary[i]
#         current_abs = predictions_abs[i]
#         testExamples[i].prediction = int(ceil(current_sign*current_abs))
#         testExamples[i].prediction = int(ceil(current_sign*average))
    
#     return predictions_abs, predictions_binary


def create_example_from_csv(row, questions, categories_dict, test=False):
    current_example = Example()
    current_example.id = row['id']
    q_id = int(row['question'])
    current_example.question = questions[q_id]
    current_example.user = int(row['user'])
    if 'position' in row:
        current_example.observation = stringToInt(row['position'])

    if not test:
        position = float(row['position'])
    else:
        position = 0

    u_id = int(row['user'])
    if u_id not in users:
        current_user = User(u_id)
        users[u_id] = current_user
        
    else:
        current_user = users[u_id]
        
    if not test: 
        if position > 0:
            correct = True
        else:
            correct = False
        answer = row['answer']
        current_user.add_question(q_id, correct, position, answer)
        
        current_example.answer = answer
        current_example.question.add_question_occurence(position)
        
        current_category = questions[q_id].category
        categories_dict[current_category].add_occurrence(current_user, position)
        
    return current_example

def recordCrossValidationResults(err, recordFile, features_abs_used, features_sign_used):
    if os.path.exists(recordFile) and os.path.isfile(recordFile):
        num_lines = sum(1 for line in open(recordFile))
    else:
        num_lines = 0
    recordFile = open(recordFile, 'a')
    writer = DictWriter(recordFile, fieldnames=["Error", "Time", "Features"])
    if num_lines == 0:
        writer.writeheader()
    
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
#     recordFile.write('\nError:  '+str(err)+' Timestamp: '+str(timestamp)+ ' Features used: '+str(features_used))
    writer.writerow({'Error':err, 'Time':timestamp, 'Features':"Continuous"+str(features_abs_used)+" Binary:"+str(features_sign_used)})
    recordFile.flush()
    recordFile.close()
    
def load_wiki_data(wiki_file):
    if os.path.exists(wiki_file) and os.path.isfile(wiki_file):
        infile = open(wiki_file, 'rb')
        print "Loading Wikipedia data"
        wiki_data = pickle.load(infile)
        infile.close()
        return wiki_data
    else:
        if not os.path.exists(wiki_file):
            print "Wikipedia data file not found"
        if os.path.exists(wiki_file) and not os.path.isfile(wiki_file):
            print "Wikipedia file exists but is not a file"
        return {}

def store_wiki_features(wiki_file, wiki_data):
    outfile = open(wiki_file, 'wb')
    print "Storing Wikipedia data"
    pickle.dump(wiki_data, outfile)
    outfile.flush()
    outfile.close()
    
def create_examples(fileName, categories_dict, questions_dict, nTrainingSets=-1, generate_test=False, all_test=False, limit=-1):
    fin = open(fileName, 'rt')
    reader = csv.DictReader(fin)
    
    train_examples = []
    test_examples = []
    
    rowIndex = 0
    nTrainingSets = args.local_selection
    total = 0.0
    count = 0.0
    for row in reader:
        if limit > 0 and rowIndex >= limit:
            break
        if all_test or generate_test and nTrainingSets > 0 and rowIndex % nTrainingSets == 0:
            current_example = create_example_from_csv(row, questions_dict, categories_dict, test=True)
            test_examples.append(current_example)
        else:
            total += float(row['position'])
            count += 1
            current_example = create_example_from_csv(row, questions_dict, categories_dict)
            train_examples.append(current_example)
        rowIndex += 1
    
    fin.close()
    
    if generate_test:
        average = total/count
        return train_examples, test_examples, average
    elif all_test:
        return test_examples
    else:
        return train_examples

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
                'dictionary':[], 'use_dictionary':False, 'unique':True, 'ngram_range':(2,20), 'user_average':True,
                'numbers':False, 'before_noun':False, 'wiki_answer':True, 'question_count':False, 'question_average':True,
                'question_percent':False, 'provided_answer':True, 'category_average':True}
    
    features_sign = {'word':False, 'speech':True, 'capital':True, 'all_upper':False, 'foreign':True, 
                'dictionary':[], 'use_dictionary':False, 'unique':True, 'ngram_range':(2,20), 'user_average':True,
                'numbers':False, 'before_noun':False, 'wiki_answer':True, 'question_count':False, 'question_average':False,
                'question_percent':False, 'provided_answer':True, 'category_average':True, 'kernel':'sigmoid'}
    
    usingWikipediaFeatures = features_abs['wiki_answer'] or features_sign['wiki_answer']
    
    if usingWikipediaFeatures and not args.regenerate_wiki:
        wiki_data_file = args.wiki_file
        wiki_data = load_wiki_data(wiki_data_file)
        original_wiki_length = len(wiki_data)
    else:
        wiki_data = {}
        original_wiki_length = 0
    
    questionFile = "questions_processed.csv"
    fin = open(questionFile, 'rt')
    reader = csv.DictReader(fin)#, ['id', 'answer', 'type', 'category', 'question', 'keywords'])
    questions = dict()
    categories = dict()
    for row in reader:
        key_dict = eval(row['keywords'])
        question_list = eval(row['question'])
        category = row['category']
        question = Question(question_list, row['category'], key_dict, row['answer'], row['type'], row['id'])
        if row['category'] not in categories:
            current_category = Category(category)
            categories[category] = current_category
        else:
            current_category = categories[category]
        current_category.add_question(question.q_id)
        questions[int(row['id'])] = question
    fin.close()
    users = {}
    # Read training examples
    trainingFile = "train.csv"
    
    trainingExamples, testExamples, average_abs = create_examples(trainingFile, categories, questions, nTrainingSets=args.local_selection, generate_test=True, limit=args.limit)
    
    
    print "\nRead data"
    
    if args.local:
        errors = []
                
        producePredictions(trainingExamples, testExamples, users, features_abs, features_sign, wiki_data, categories, average_abs) 
        
        print "\nFinished prediction"
        
        err = rootMeanSquaredError(testExamples)
        
        recordCrossValidationResults(err, 'records.csv', features_abs, features_sign)
        
        print "CROSS-VALIDATION RESULTS"
        print "ERROR: ", err#np.mean(errors)

    if args.predict:
        # Read test file
        print "\nWriting predictions file"
        testFile = "test.csv"
        
        testExamples = create_examples(testFile, categories, questions, all_test=True)
        
        trainingExamples = create_examples(trainingFile, categories, questions)
        
        # Generate predictions
        producePredictions(trainingExamples, testExamples, users, features_abs, features_sign, wiki_data, categories, average_abs)
     
        # Produce submission file
        submissionFile = "submission.csv"
        fout = open(submissionFile, 'w')
        writer = csv.writer(fout)
        writer.writerow(("id","position"))
        for ex in testExamples:
            writer.writerow((ex.id, ex.prediction))
        fout.close()
        print "\nPredictions file written"
        
    if usingWikipediaFeatures:
        wiki_data_file = args.wiki_file
        if args.regenerate_wiki or len(wiki_data) > 0 and original_wiki_length != len(wiki_data):
            store_wiki_features(wiki_data_file, wiki_data)


