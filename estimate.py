
import argparse
import csv
import nltk
import re
from sets import Set
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet as wn
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import mean_squared_error
import sys

import numpy as np


VALID_POS = ['FW', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNS', 
             'NNP', 'NNPS', 'PDT', 'POS', 'RB', 'RBR', 'RBS',
             'RP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WRB']

class Analyzer:
    def __init__(self, word=False, speech=False, capital=False, all_upper=False, foreign=False, unique=False, dictionary=dict(), use_dictionary=False, ngram_range=(1,1)):
        self.word = word
        self.speech = speech
        self.capital = capital
        self.ngram_range = ngram_range
        self.all_upper = all_upper
        self.foreign = foreign
        self.dictionary = dictionary
        self.use_dictionary = use_dictionary
        self.unique=unique
    
    def strip_tags_from_words(self, word_list):
        stripped_words = []
        for word_feature in word_list:
            stripped_words.append(word_feature.split(":")[0])
        return stripped_words
    
    def __call__(self, feature_string):
        feats = feature_string.split()
        
        if self.word and self.ngram_range != (1,1):
            current_words = self.strip_tags_from_words(feats)
            for ngram_length in range(self.ngram_range[0], self.ngram_range[1]+1):
                    for ngram in zip(*[current_words[i:] for i in range(3)]):
                        ngram = " ".join(ngram)
                        yield ngram
#                     for j in range(self.ngram_range[0], self.ngram_range[1]):
        
        for feat_tuple in feats:
            feat_tokens = feat_tuple.split(":")
            
            if self.word: #and self.ngram_range == (1,1):
                yield feat_tokens[0]
                
            if self.speech:
                if feat_tokens[1] in VALID_POS:
                    yield feat_tokens[1]
                    
            if self.capital:
                if feat_tokens[2] == "C":
                    yield feat_tokens[2]
                    
            if self.all_upper:
                if feat_tokens[3] == "U":
                    yield feat_tokens[3]
                    
            if self.foreign:
                if feat_tokens[4] == "F":
                    yield feat_tokens[4]
                    
            if self.unique:
                if feat_tokens[5] == "UN":
                    yield feat_tokens[5]
            
#             if self.key:
#                     if feat_tokens[6] == "KEY":
#                         yield feat_tokens[6]
                    
            if self.use_dictionary:
                for category in self.dictionary:
                    if feat_tokens[0] in self.dictionary[category]:
                        yield category.upper()

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
        
class Featurizer:
    def __init__(self, use_default=False, analyzer=u'word'):
        self.stop_words=['uner', 'via', 'answer', 'ANSWER', 'was', 'his','the','that','and', 'points', 'pointsname', 'this', 'for', 'who', 'they','can', 'its', 'also', 'these', 'are', 'name', 'after', 'than', 'had', 'with', 'about', 'ftp', '10', '15', 'this', 'from', 'not', 'has', 'well']
        if use_default:
#             self.stop_words=['was', 'his','the','that','and', 'points', 'pointsname', 'this', 'for', 'who', 'one', 'two', 'three', 'four', 'ten', 'they','can', 'its']
#             self.vectorizer = CountVectorizer(token_pattern=u'(?u)\\b[a-zA-Z]{2}[a-zA-Z]+|[0-9]{4}\\b', ngram_range=(1,7), stop_words=self.stop_words)
            self.vectorizer = CountVectorizer(ngram_range=(1,20), stop_words=self.stop_words)
        else:
#             self.
#             self.vectorizer = CountVectorizer(token_pattern=u'(?u)\\b[a-zA-Z]{2}[a-zA-Z]+|[0-9]{4}\\b', stop_words=['was', 'his','the','that','and', 'points', 'pointsname', 'this', 'for', 'who', 'they','can', 'its', 'also', 'these', 'are', 'name', 'after', 'than', 'had', 'with', 'about', 'ftp', '10', '15']ngram_range=(1,7), analyzer = analyzer, stop_words=self.stop_words)
            self.vectorizer = CountVectorizer(analyzer=analyzer)


    def train_feature(self, examples, limit=-1):
#         return self.vectorizer.fit_transform(examples)
        return self.vectorizer.fit_transform(ex for ex in all_examples(limit, examples))

    def test_feature(self, examples, limit=-1):
#         return self.vectorizer.transform(examples)
        return self.vectorizer.transform(ex for ex in all_examples(limit, examples))

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

def all_examples(limit, examples):
    
    for current_example in examples:
#         example_str = ""
#         example_str += "CAT:"+current_example.question.category
#         example_str += " "
#         example_str += current_example.question.question
        tagged_example_str = tagged_example(current_example.question.question, current_example.question.category, current_example.question.keywords, current_example.question.answer) 
        yield tagged_example_str

def tagged_example(question, category, keywords, answer):
#     print keywords
    annotated_words = []
    stop_words=['uner', 'via', 'answer', 'ANSWER', 'was', 'his','the','that','and', 'points', 'pointsname', 'this', 'for', 'who', 'they','can', 'its', 'also', 'these', 'are', 'name', 'after', 'than', 'had', 'with', 'about', 'ftp', '10', '15', 'this', 'from', 'not', 'has', 'well']
    prev_word = ""
    prev_word_unannotated = ""
    reg_exp = re.compile(u'(?u)\\b[a-zA-Z]{2}[a-zA-Z]+|[0-9]{4}\\b')
    reg_exp_alphanum = re.compile(u'\\w')
#     for key in keywords:#question.split():
#         word = keywords[key]
    words = question.split()
    words.append(answer);
    for key in keywords:
        words.append(keywords[key])
    for word in words:
#         print word
        result = reg_exp.match(word)
        if result is None:
            continue
        original_annotated_word = result.string[result.start(0):result.end(0)]
        annotated_word = original_annotated_word
        if original_annotated_word.lower() in stop_words:
            prev_word_unannotated = word
            continue
        annotated_word += ":"
#        TODO: this is where POS classification would go
        annotated_word += ":"
        if original_annotated_word[0].isupper() and len(prev_word_unannotated) > 0 and reg_exp_alphanum.match(prev_word_unannotated[len(prev_word_unannotated)-1]):
            annotated_word+="C"
        annotated_word += ":"
        if original_annotated_word.isupper():
            annotated_word+="U"
        annotated_word += ":"
        if not wn.synsets(original_annotated_word):
            annotated_word+="F"
        annotated_word += ":"    
#         sum_t = sum(1 for x in keywords.values() if(word == x))
#         print str(word) + " " + str(sum_t)
#         if sum_t == 1:
        if words.count(original_annotated_word) == 1:
            annotated_word+="UN"
        annotated_words.append(annotated_word)
        prev_word = original_annotated_word
        prev_word_unannotated = word

        
#     tokenized_words = nltk.tokenize.word_tokenize(words)
#     print 'tokenized iteration: '+ str(current_example.counter)
#     current_example.counter += 1
#     tagged_words = nltk.pos_tag(tokenized_words)
#     tagged_line = ""
#     for word, tag in tagged_words:
#         tagged_line += (word + ":" +tag + " ")
#     return tagged_line
#     return ""
    tagged = " ".join(annotated_words)
#     print tagged
    return tagged

def stringToInt(s):
    return int(float(s))

def rootMeanSquaredError(examples):
    predictions = [x.prediction for x in examples]
    observations = [x.observation for x in examples]
    return mean_squared_error(predictions, observations) ** 0.5

def producePredictions(trainingExamples, testExamples):
    analyzer = Analyzer(word=True, speech=False, capital=True, all_upper=True, foreign=True, dictionary=[], use_dictionary=False, unique=False, ngram_range=(1,4))
    featurizer = Featurizer(use_default=False, analyzer=analyzer)
    x_train = featurizer.train_feature(trainingExamples)
    x_test = featurizer.test_feature(testExamples)
    y_train = []
    for train_example in trainingExamples:
        y_train.append(train_example.observation)
    
#     lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
#     
#     lr.fit(x_train, y_train)

    linreg = linear_model.LinearRegression()  
    linreg.fit(x_train, y_train)  
    
    print "\nFit training data"
    
#     predictions = lr.predict(x_test)
    predictions = linreg.predict(x_test)
    
    for i in range(len(predictions)):
        testExamples[i].prediction = predictions[i]
    
    return predictions
#     # Calculate mean and standard deviation of training examples
#     averageTime = np.mean([x.observation for x in trainingSet])
#     stdDev = np.std([x.observation for x in trainingSet])
# 
#     # Produce predictions
#     for current_example in testSet:
#         current_example.prediction = averageTime

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
    
    questionFile = "questions.csv"
    fin = open(questionFile, 'rt')
    reader = csv.DictReader(fin, ['id', 'answer', 'type', 'category', 'question', 'keywords'])
    questions = dict()
    for row in reader:
        key_dict = eval(row['keywords'])
#         for key in key_dict:
#             print key_dict[key]
        question = Question(row['question'], row['category'], key_dict, row['answer'], row['type'], row['id'])
        questions[int(row['id'])] = question
    fin.close()
    
    # Read training examples
    trainingFile = "train.csv"
    fin = open(trainingFile, 'rt')
    reader = csv.DictReader(fin)
    examples = []
    for row in reader:
        current_example = Example()
        current_example.id = row['id']
        current_example.question = questions[int(row['question'])]
        current_example.user = row['user']
        current_example.observation = stringToInt(row['position'])
        current_example.answer = row['answer']
        examples.append(current_example)
    fin.close()
    
    print "\nRead data"
    
    #    current_example.observation = (current_example.observation - mean)/dev

    # Define cross-validation parameters
    if args.local:
        nTrainingSets = args.local_selection
        trainingExamples = []
        testExamples = []
        
        for i in range(len(examples)):
            if i%nTrainingSets == 0:
                testExamples.append(examples[i])
            else:
                trainingExamples.append(examples[i])
                
        predictions = producePredictions(trainingExamples, testExamples) 
        
        
        print "\nFinished prediction"
        
        for i in range(len(predictions)):
            predict_y = predictions[i]
            test_example = testExamples[i]
            test_example.prediction = predict_y
        #     boundaryIndices = [int(x) for x in np.linspace(0, len(examples)-1, nTrainingSets+1)]
        #     trainingSets = []
        #     for i in range(len(boundaryIndices)-1):
        #         set_i = examples[boundaryIndices[i] : boundaryIndices[i+1]]
        #         trainingSets.append(set_i)
        
        # Perform cross validation on training examples 
        #     errors = []
        #     for i in range(nTrainingSets):
        #         # Partition training examples into train and validation sets
        #         trainingExamples = []
        #         verificationExamples = trainingSets[i]
        #         for j in range(nTrainingSets):
        #             if j != i:
        #                 trainingExamples = trainingExamples + trainingSets[j]
        # 
        #         # Generate predictions
        #         producePredictions(trainingExamples, verificationExamples)
        # 
        #         # Calculate root-mean-square deviation
        err = rootMeanSquaredError(testExamples)
        #         errors.append(err)
        
        print "CROSS-VALIDATION RESULTS"
        print "ERROR: ", err#np.mean(errors)
        #print "min median max:   ", min(errors), "  ", np.median(errors), "  ", max(errors)

    if args.predict:
        # Read test file
        print "\nWriting predictions file"
        testFile = "test.csv"
        fin = open(testFile, 'rt')
        reader = csv.DictReader(fin)
        testExamples = []
        for row in reader:
            current_example = Example()
            current_example.id = row['id']
            current_example.question = questions[int(row['question'])]
            current_example.user = row['user']
            testExamples.append(current_example)
        fin.close()
     
        # Generate predictions
        producePredictions(examples, testExamples)
     
        # Produce submission file
        submissionFile = "submission.csv"
        fout = open(submissionFile, 'w')
        writer = csv.writer(fout)
        writer.writerow(("id","position"))
        for ex in testExamples:
            writer.writerow((ex.id, ex.prediction))
        fout.close()
        print "\nPredictions file written"


