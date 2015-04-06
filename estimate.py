
import argparse
import csv
from sets import Set
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import mean_squared_error
import sys

import numpy as np


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
        if use_default:
#             self.stop_words=['was', 'his','the','that','and', 'points', 'pointsname', 'this', 'for', 'who', 'one', 'two', 'three', 'four', 'ten', 'they','can', 'its']
#             self.vectorizer = CountVectorizer(token_pattern=u'(?u)\\b[a-zA-Z]{2}[a-zA-Z]+|[0-9]{4}\\b', ngram_range=(1,7), stop_words=self.stop_words)
            self.vectorizer = CountVectorizer()
        else:
#             self.
#             self.vectorizer = CountVectorizer(token_pattern=u'(?u)\\b[a-zA-Z]{2}[a-zA-Z]+|[0-9]{4}\\b', stop_words=['was', 'his','the','that','and', 'points', 'pointsname', 'this', 'for', 'who', 'they','can', 'its', 'also', 'these', 'are', 'name', 'after', 'than', 'had', 'with', 'about', 'ftp', '10', '15']ngram_range=(1,7), analyzer = analyzer, stop_words=self.stop_words)
            self.vectorizer = CountVectorizer(analyzer=analyzer, ngram_range=(1,7))


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
    for example in examples:
        example_str = ""
#         example_str += "CAT:"+example.question.category
#         example_str += " "
        example_str += example.question.question
        
        yield example_str

def stringToInt(s):
    return int(float(s))

def rootMeanSquaredError(examples):
    predictions = [x.prediction for x in examples]
    observations = [x.observation for x in examples]
    return mean_squared_error(predictions, observations) ** 0.5

def producePredictions(trainingExamples, testExamples):
    featurizer = Featurizer(use_default=True)
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
#     for example in testSet:
#         example.prediction = averageTime

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
    args = argparser.parse_args()
    
    questionFile = "questions.csv"
    fin = open(questionFile, 'rt')
    reader = csv.DictReader(fin, ['id', 'answer', 'type', 'category', 'question', 'keywords'])
    questions = dict()
    for row in reader:
        question = Question(row['question'], row['category'], row['keywords'], row['answer'], row['type'], row['id'])
        questions[int(row['id'])] = question
    fin.close()
    
    # Read training examples
    trainingFile = "train.csv"
    fin = open(trainingFile, 'rt')
    reader = csv.DictReader(fin)
    examples = []
    for row in reader:
        example = Example()
        example.id = row['id']
        example.question = questions[int(row['question'])]
        example.user = row['user']
        example.observation = stringToInt(row['position'])
        example.answer = row['answer']
        examples.append(example)
    fin.close()
    
    print "\nRead data"

    #    example.observation = (example.observation - mean)/dev

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
            example = Example()
            example.id = row['id']
            example.question = questions[int(row['question'])]
            example.user = row['user']
            testExamples.append(example)
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


