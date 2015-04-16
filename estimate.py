
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
import numpy
import os
import pickle
import re
from scipy.sparse.csgraph._min_spanning_tree import csr_matrix
from sets import Set
from sklearn import linear_model
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

# question_features = {}
# user_features = {}
# answer_wiki_features = {}

class Analyzer:
    def __init__(self, features):#word=False, speech=False, capital=False, all_upper=False, foreign=False, 
#                  unique=False, dictionary=dict(), use_dictionary=False, ngram_range=(1,1), user_average=False):
#         self.word = word
#         self.speech = speech
#         self.capital = capital
#         self.ngram_range = ngram_range
#         self.all_upper = all_upper
#         self.foreign = foreign
#         self.dictionary = dictionary
#         self.use_dictionary = use_dictionary
#         self.unique=unique
#         self.user_average = user_average
        self.features = defaultdict(lambda: False)
        for feature in features:
            self.features[feature] = features[feature]
        self.numeric_features = defaultdict(dict)
        self.index = 0
    
    def strip_tags_from_words(self, word_list):
        stripped_words = []
        for word_feature in word_list:
            stripped_words.append(word_feature.split(":")[0])
        return stripped_words
    
    def store_numeric_feature(self, index, feat_index, numeric_feature):
        self.numeric_features[index][feat_index] = float(numeric_feature)
    
    def add_numeric_features(self, feature_matrix):
#         print type(feature_matrix)
        feature_array = feature_matrix.toarray()
#         print feature_matrix
        new_feature_array = []
        for i in range(len(feature_array)):
#             print i
            current_features = feature_array[i].tolist()
#             print current_features
#             for j in self.numeric_features[i]:
#             current_feats = []
            for feat_index in self.numeric_features[i]:
                current_features.append(self.numeric_features[i][feat_index])
#             new_array = numpy.append(current_features, current_feats)
#             current_feats.append(current_feats)
#             print new_array
#             new_feature_array.append(new_array)
            new_feature_array.append(current_features)
#             print current_features
#         print new_feature_array
        return csr_matrix(new_feature_array)
    
    def __call__(self, feature_string):
#         print feature_string
        feature_strings_ = feature_string.split("#")
        feat_id = int(feature_strings_[0])
        feature_strings = feature_strings_[1].split("|")
#         print feature_strings
        user_features = feature_strings[0]
#         print user_features
        question_features = feature_strings[1].split()
        
        answer_features = feature_strings[2]
#         print question_features
#         print self.features
        if self.features['ngram_range'] != (1,1):
            current_words = self.strip_tags_from_words(question_features)
            for ngram_length in range(self.features['ngram_range'][0], self.features['ngram_range'][1]+1):
                    for ngram in zip(*[current_words[i:] for i in range(3)]):
                        ngram = " ".join(ngram)
                        yield ngram
        
        for feat_tuple in question_features:
            feat_tokens = feat_tuple.split(":")
#             print feat_tuple
            if self.features['word']: #and self.ngram_range == (1,1):
                yield feat_tokens[0]
                
            if self.features['speech']:
                if feat_tokens[1] != "Unk":
#                     print feat_tokens[1]
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
                self.store_numeric_feature(feat_id, 0, int(feat_tokens[8]))
            
            if self.features['question_average']:
                self.store_numeric_feature(feat_id, 1, float(feat_tokens[9]))
                
            if self.features['question_percent']:
                self.store_numeric_feature(feat_id, 2, float(feat_tokens[10]))
            
#             if self.key:
#                     if feat_tokens[6] == "KEY":
#                         yield feat_tokens[6]
                    
            if self.features['use_dictionary']:
                for category in self.dictionary:
                    if feat_tokens[0] in self.dictionary[category]:
                        yield category.upper()
        
        user_features_tokens = user_features.split(":")
#             user_feat_tokens = user_feature_tuple.split(":")
#         print user_features_tokens
        if self.features['user_average']:
#                 print user_feat_tokens
            if user_features_tokens[0] == "USR_AVG":
#                 print "here"
                self.store_numeric_feature(feat_id, 3, float(user_features_tokens[1]))
#                 for i in range(int(float(user_features_tokens[1]))):
#                     yield "USR_AVG"
            else:
                self.store_numeric_feature(feat_id, 3, 0)
                    
        answer_features_tokens = answer_features.split(":")
#         print answer_features_tokens
        if self.features['wiki_answer']:
#             if answer_features_tokens[0] == "WIKI_NUM":
#                 for i in range(int(answer_features_tokens[1])):
#                     yield "WIKI_NUM"
            if answer_features_tokens[2] == "WIKI_FIRST":
                self.store_numeric_feature(feat_id, 4, int(answer_features_tokens[3]))
#                 print int(answer_features_tokens[3])
#                 for i in range(int(answer_features_tokens[3])):
#                     yield "WIKI_FIRST"
            else:
                self.store_numeric_feature(feat_id, 4, 0)
#         self.index+=1
        if self.features['provided_answer']:
#             print feat_tokens
#             print "here"+feat_tokens[3]+" "+feat_tokens[4]
            self.store_numeric_feature(feat_id, 5, int(answer_features_tokens[4]))
        else:
            self.store_numeric_feature(feat_id, 5, 0)

class Example:
    def __init__(self):
        self.id = 0
        self.question = ""
        self.user = 0
        self.answer = ""
        self.observation =  0
        self.prediction = 0


        
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
    def __init__(self, features=defaultdict(lambda: False), use_default=False, analyzer=u'word'):
        self.stop_words=['uner', 'via', 'answer', 'ANSWER', 'was', 'his','the','that','and', 'points', 'pointsname', 'this', 'for', 'who', 'they','can', 'its', 'also', 'these', 'are', 'name', 'after', 'than', 'had', 'with', 'about', 'ftp', '10', '15', 'this', 'from', 'not', 'has', 'well']
        self.analyzer = analyzer
        
        if use_default:
            self.vectorizer = CountVectorizer(ngram_range=(1,10), stop_words=self.stop_words)
            self.default = True
            self.features = features
        else:
            self.vectorizer = CountVectorizer(analyzer=analyzer)
            self.default = False
            self.features = features


    def train_feature(self, examples, users, wiki_data, limit=-1):
#         return self.vectorizer.fit_transform(examples)
        count_features = self.vectorizer.fit_transform(ex for ex in all_examples(limit, examples, users, self.features, wiki_data, default = self.default))
        if not self.default:
            return self.analyzer.add_numeric_features(count_features)
#         features = numpy.array
#         features = []
#         for ex in examples:
#             current_features = []
#             for feature in all_examples(limit, [ex], users, self.features, default = self.default):
#                 current_features.append(feature)
#             features.append(current_features)
#         print features
        return count_features
            

    def test_feature(self, examples, users, wiki_data, limit=-1):
#         return self.vectorizer.transform(examples)
        count_features = self.vectorizer.transform(ex for ex in all_examples(limit, examples, users, self.features, wiki_data, default = self.default))
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



def all_examples(limit, examples, users, features, answer_wiki_features, default=False):
    
    for i in range(len(examples)):
        current_example = examples[i]
#         example_str = ""
#         example_str += "CAT:"+current_example.question.category
#         example_str += " "
#         example_str += current_example.question.question
        current_user = users[current_example.user]
#         print current_example.question
#         print current_example.observation
        tagged_example_str = get_example_features(i,features, current_example.question, current_user, answer_wiki_features, default) 
        yield tagged_example_str


        
def get_user_features(features, user):
#     if user.u_id in user_features:
#         return user_features[user.u_id]
    
    average = str(abs(user.average_position))
#     print average
    user_feature = ""
    if features['user_average']:
#         print average
        user_feature += "USR_AVG:"+average 
#     if user.u_id not in user_features:
#         user_features[user.u_id] = user_feature 
    return user_feature


        
def get_question_features(features, question_container):
#     print features
#     print question_container
    question = question_container.question
    category = question_container.category
    keywords = question_container.keywords
    answer = question_container.answer
    question_id = question_container.q_id
#     if question_id in question_features:
#         return question_features[question_id]
    
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
    
#     words[word_index] = answer
#     word_index = word_index + 1

#     for key in keywords:
#         words[word_index] = keywords[key]
#         word_index = word_index + 1

    encounteredNoun = False
        
    for word_index in words:
        word = words[word_index]
        result = reg_exp.match(word)
        if result is None:
            continue
        original_annotated_word = result.string[result.start(0):result.end(0)]
        annotated_word = ""
        if features['word']:
#             print 'here'
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
            annotated_word += str(question_container.count)
        annotated_word += ":"
        if features['question_average']:
            annotated_word += str(question_container.average_response)
        annotated_word += ":"
        if features['question_percent']:
            annotated_word += str(question_container.absolute_average)
        
        
        annotated_words.append(annotated_word)
        prev_word = original_annotated_word
        prev_word_unannotated = word
        
#     annotated_words += "|"
#     if features['wiki_answer']:
#         if answer in answer_wiki_features:
#             num_results = answer_wiki_features[answer][0]
#             first_len = answer_wiki_features[answer][1]
#             annotated_words += "WIKI_NUM:"+str(num_results)+":WIKI_FIRST:"+str(first_len)
#         else:
#             try:
#                 results = eval(str(wikipedia.search(answer)))
#                 num_results = len(results)
#                 print num_results
#                 print results
#                 annotated_words += "WIKI_NUM:"+str(num_results)+":WIKI_FIRST:"
#                 if num_results > 0:
#                     first_result = wikipedia.page(str(results[0]))
#                     first_len = len(first_result.content)
#                     annotated_words+=str(first_len)
#                     print first_len
#                     answer_wiki_features[answer] = [num_results,first_len]
#                 else:
#                     annotated_words+="0"
#                     answer_wiki_features[answer] = [num_results,0]
#             except:
#                 annotated_words += "WIKI_NUM:0:WIKI_FIRST:0"
#                 answer_wiki_features[answer] = [0,0]
#     else:
#         annotated_words += ":::"
#     annotated_words += ":"
    

        
    tagged = " ".join(annotated_words)
#     question_features[question_id] = tagged
#     print tagged
    return tagged

def get_answer_features(answer, answer_wiki_features, provided_answer):
    annotated_words = ""
    question_string = "'"
    incorrect_threshold = 10
    if features['wiki_answer']:
        if answer in answer_wiki_features:
            num_results = answer_wiki_features[answer][0]
            first_len = answer_wiki_features[answer][1]
            num_occurences = answer_wiki_features[answer][2]
#             print num_occurences
            question_string += "WIKI_NUM:"+str(num_results)+":WIKI_FIRST:"+str(first_len)+":"+str(num_occurences)
        else:
            try:
                results, suggestion = eval(str(wikipedia.search(answer, results=1000000, suggestion=True)))
                if suggestion is not None:
                    print "Suggestion: "+str(suggestion)+" "+str(len(results))+" answer: "+answer
                    if len(results) <= incorrect_threshold:
                        results = eval(str(wikipedia.search(str(suggestion), results=1000000)))
#                 print results
                num_results = len(results)
#                 print num_results
#                 print results
                question_string += "WIKI_NUM:"+str(num_results)+":WIKI_FIRST:"
                if num_results > 0:
                    first_result = wikipedia.page(str(results[0]))
                    first_len = len(first_result.content)
                    question_string+=str(first_len)
#                     print first_len
                    
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
#     print question_string
    return annotated_words


def get_example_features(example_id, features, question, user, answer_wiki_features, default=False):
    
    question_features = get_question_features(features, question)
    user_features = get_user_features(features, user)
    answer = question.answer
    question_id = question.q_id
    user_questions = user.questions
    if question_id in user_questions:
        provided_answer = user.questions[question.q_id]['answer']
    else:
        provided_answer = ""
    answer_features = get_answer_features(answer, answer_wiki_features, provided_answer)
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
        feature_string = str(example_id)+"#"+user_features +"|"+question_features+"|"+answer_features
    return feature_string


def stringToInt(s):
    return int(float(s))


def rootMeanSquaredError(examples):
    predictions = [x.prediction for x in examples]
    observations = [x.observation for x in examples]
    return mean_squared_error(predictions, observations) ** 0.5


def producePredictions(trainingExamples, testExamples, users, features, wiki_data):
    analyzer = Analyzer(features=features)
    featurizer = Featurizer(features, use_default=False, analyzer=analyzer)
#     featurizer = Featurizer(use_default=True)
    y_train = []
    for train_example in trainingExamples:
        y_train.append(train_example.observation)
    print "Generating training x"
    x_train = featurizer.train_feature(trainingExamples, users, wiki_data)
    print featurizer.vectorizer.vocabulary_.get('document')
#     for feat in x_train:
#         print feat
    print x_train.toarray()[0]
    print x_train.toarray()[50]
#     print x_train.toarray()[4000]
    del trainingExamples
    print "Generating test x"
    x_test = featurizer.test_feature(testExamples, users, wiki_data)
    
    
#     classifier = SGDClassifier(loss='log', penalty='l2', shuffle=True)
#     
#     lr.fit(x_train, y_train)

    classifier = linear_model.LinearRegression()
    
    print "Fitting classifier"  
    classifier.fit(x_train, y_train)  
    
    print "\nFit training data"
    
    predictions = classifier.predict(x_test)
    
    for i in range(len(predictions)):
        testExamples[i].prediction = int(ceil(predictions[i]))
    
    return predictions


def create_example_from_csv(row, questions, test=False):
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
    return current_example

def recordCrossValidationResults(err, recordFile, features_used):
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
    writer.writerow({'Error':err, 'Time':timestamp, 'Features':str(features_used)})
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
    
def create_examples(fileName, nTrainingSets=-1, generate_test=False, all_test=False, limit=-1):
    fin = open(fileName, 'rt')
    reader = csv.DictReader(fin)
    
    train_examples = []
    test_examples = []
    
    rowIndex = 0
    nTrainingSets = args.local_selection
    for row in reader:
        if limit > 0 and rowIndex >= limit:
            break
        if all_test or generate_test and nTrainingSets > 0 and rowIndex % nTrainingSets == 0:
            current_example = create_example_from_csv(row, questions, test=True)
            test_examples.append(current_example)
        else:
            current_example = create_example_from_csv(row, questions)
            train_examples.append(current_example)
        rowIndex += 1
    
    fin.close()
    if generate_test:
        return train_examples, test_examples
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
    
    features = {'word':False, 'speech':True, 'capital':True, 'all_upper':False, 'foreign':True, 
                'dictionary':[], 'use_dictionary':False, 'unique':True, 'ngram_range':(2,10), 'user_average':True,
                'numbers':False, 'before_noun':True, 'wiki_answer':True, 'question_count':False, 'question_average':False,
                'question_percent':False, 'provided_answer':False}
    
    usingWikipediaFeatures = features['wiki_answer']
    
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
    for row in reader:
        key_dict = eval(row['keywords'])
        question_list = eval(row['question'])
        question = Question(question_list, row['category'], key_dict, row['answer'], row['type'], row['id'])
        questions[int(row['id'])] = question
    fin.close()
    users = {}
    # Read training examples
    trainingFile = "train.csv"
    
    trainingExamples, testExamples = create_examples(trainingFile, nTrainingSets=args.local_selection, generate_test=True, limit=args.limit)
    
    
    print "\nRead data"
    
    if args.local:
        errors = []
                
        predictions = producePredictions(trainingExamples, testExamples, users, features, wiki_data) 
        
        print "\nFinished prediction"
        
#         for i in range(len(predictions)):
#             predict_y = predictions[i]
#             test_example = testExamples[i]
#             test_example.prediction = predict_y
    
        # Calculate root-mean-square deviation
        err = rootMeanSquaredError(testExamples)
        
#         print "Error for iteration "+str(iteration)+": "+str(err)
#         errors.append(err)
        
#         avg_error = np.mean(errors)
        
        recordCrossValidationResults(err, 'records.csv', features)
        
        print "CROSS-VALIDATION RESULTS"
        print "ERROR: ", err#np.mean(errors)

    if args.predict:
        # Read test file
        print "\nWriting predictions file"
        testFile = "test.csv"
        
        testExamples = create_examples(testFile, all_test=True)
        
        trainingExamples = create_examples(trainingFile)
        
        # Generate predictions
        producePredictions(trainingExamples, testExamples, users, features, wiki_data)
     
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


