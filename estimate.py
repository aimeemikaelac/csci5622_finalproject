
from _collections import defaultdict
import argparse
from collections import Counter
import csv
import nltk
from nltk.corpus import wordnet as wn
import numpy
import re
from sets import Set
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import mean_squared_error
import sys

import numpy as np


VALID_POS = ['FW', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 
             'NNP', 'NNPS', 'PDT', 'POS', 'RB', 'RBR', 'RBS',
             'RP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WRB']

# question_features = {}
# user_features = {}

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
    
    def strip_tags_from_words(self, word_list):
        stripped_words = []
        for word_feature in word_list:
            stripped_words.append(word_feature.split(":")[0])
        return stripped_words
    
    def __call__(self, feature_string):
        feature_strings = feature_string.split("|")
#         print feature_strings
        user_features = feature_strings[0]
#         print user_features
        question_features = feature_strings[1].split()
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
                for i in range(int(float(user_features_tokens[1]))):
                    yield "USR_AVG"



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
        
        if use_default:
            self.vectorizer = CountVectorizer(ngram_range=(1,10), stop_words=self.stop_words)
            self.default = True
            self.features = features
        else:
            self.vectorizer = CountVectorizer(analyzer=analyzer)
            self.default = False
            self.features = features


    def train_feature(self, examples, users, limit=-1):
#         return self.vectorizer.fit_transform(examples)
        count_features = self.vectorizer.fit_transform(ex for ex in all_examples(limit, examples, users, self.features, default = self.default))
#         features = numpy.array
#         features = []
#         for ex in examples:
#             current_features = []
#             for feature in all_examples(limit, [ex], users, self.features, default = self.default):
#                 current_features.append(feature)
#             features.append(current_features)
#         print features
        return count_features
            

    def test_feature(self, examples, users, limit=-1):
#         return self.vectorizer.transform(examples)
        return self.vectorizer.transform(ex for ex in all_examples(limit, examples, users, self.features, default = self.default))

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



def all_examples(limit, examples, users, features, default=False):
    
    for current_example in examples:
#         example_str = ""
#         example_str += "CAT:"+current_example.question.category
#         example_str += " "
#         example_str += current_example.question.question
        current_user = users[current_example.user]
#         print current_example.question
#         print current_example.observation
        tagged_example_str = get_example_features(features, current_example.question, current_user, default) 
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
            annotated_word += pos[word_index]
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
        annotated_words.append(annotated_word)
        prev_word = original_annotated_word
        prev_word_unannotated = word

        
    tagged = " ".join(annotated_words)
#     question_features[question_id] = tagged
#     print tagged
    return tagged


def get_example_features(features, question, user, default=False):
    
    question_features = get_question_features(features, question)
    user_features = get_user_features(features, user)
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
        feature_string = " ".join(feature_list)
    else:
        feature_string = user_features +"|"+question_features
    return feature_string


def stringToInt(s):
    return int(float(s))


def rootMeanSquaredError(examples):
    predictions = [x.prediction for x in examples]
    observations = [x.observation for x in examples]
    return mean_squared_error(predictions, observations) ** 0.5


def producePredictions(trainingExamples, testExamples, users):
    features = {'word':False, 'speech':True, 'capital':True, 'all_upper':False, 'foreign':True, 
                'dictionary':[], 'use_dictionary':False, 'unique':False, 'ngram_range':(2,10), 'user_average':True}
    analyzer = Analyzer(features=features)
    featurizer = Featurizer(features, use_default=False, analyzer=analyzer)
#     featurizer = Featurizer(use_default=True)
    y_train = []
    for train_example in trainingExamples:
        y_train.append(train_example.observation)
    print "Generating training x"
    x_train = featurizer.train_feature(trainingExamples, users)
    print featurizer.vectorizer.vocabulary_.get('document')
    print x_train.toarray()[0]
    print x_train.toarray()[1]
    print x_train.toarray()[4000]
    del trainingExamples
    print "Generating test x"
    x_test = featurizer.test_feature(testExamples, users)
    
    
#     classifier = SGDClassifier(loss='log', penalty='l2', shuffle=True)
#     
#     lr.fit(x_train, y_train)

    classifier = linear_model.LinearRegression()
    
    print "Fitting classifier"  
    classifier.fit(x_train, y_train)  
    
    print "\nFit training data"
    
    predictions = classifier.predict(x_test)
    
    for i in range(len(predictions)):
        testExamples[i].prediction = predictions[i]
    
    return predictions


def create_example_from_csv(row, questions, test=False):
    current_example = Example()
    current_example.id = row['id']
    q_id = int(row['question'])
    current_example.question = questions[q_id]
    current_example.user = int(row['user'])
    
    if not test:
        current_example.observation = stringToInt(row['position'])
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
    return current_example



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
    args = argparser.parse_args()
    
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
    fin = open(trainingFile, 'rt')
    reader = csv.DictReader(fin)
    examples = []
    for row in reader:
        current_example = create_example_from_csv(row, questions)
        examples.append(current_example)
    fin.close()
    
    print "\nRead data"
    
    # Define cross-validation parameters
    limit = args.limit
    
    if args.local:
        nTrainingSets = args.local_selection
        trainingExamples = []
        testExamples = []
        
        for i in range(len(examples)):
            if limit > 0 and i >= limit:
                break
            if i%nTrainingSets == 0:
                testExamples.append(examples[i])
            else:
                trainingExamples.append(examples[i])
                
        predictions = producePredictions(trainingExamples, testExamples, users) 
        
        
        print "\nFinished prediction"
        
        for i in range(len(predictions)):
            predict_y = predictions[i]
            test_example = testExamples[i]
            test_example.prediction = predict_y

        # Calculate root-mean-square deviation
        err = rootMeanSquaredError(testExamples)
        
        print "CROSS-VALIDATION RESULTS"
        print "ERROR: ", err#np.mean(errors)

    if args.predict:
        # Read test file
        print "\nWriting predictions file"
        testFile = "test.csv"
        fin = open(testFile, 'rt')
        reader = csv.DictReader(fin)
        testExamples = []
        for row in reader:
            current_example = create_example_from_csv(row, questions, test=True)
            testExamples.append(current_example)
        fin.close()
     
        # Generate predictions
        producePredictions(examples, testExamples, users)
     
        # Produce submission file
        submissionFile = "submission.csv"
        fout = open(submissionFile, 'w')
        writer = csv.writer(fout)
        writer.writerow(("id","position"))
        for ex in testExamples:
            writer.writerow((ex.id, ex.prediction))
        fout.close()
        print "\nPredictions file written"


