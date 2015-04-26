from csv import DictWriter
import csv
import datetime
from math import ceil
from numpy import sign
import os
import pickle
import random
from sklearn import ensemble, svm, neighbors
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics.regression import mean_squared_error
import time

from Analyzer import Analyzer
from Category import Category
from Example import Example
from Featurizer import Featurizer
from Question import Question
from User import User


class Predictor:
    def __init__(self, input_args, position_features, correctness_features, n_estimators=650, cluster=False):
        self.args = input_args
        self.position_features = position_features
        self.correctness_features = correctness_features
        self.n_estimators = n_estimators
        self.cluster = False
    
    def stringToInt(self, s):
        return int(float(s))  
      
        
    def rootMeanSquaredError(self, examples):
        predictions = [x.prediction for x in examples]
        observations = [x.observation for x in examples]
        return mean_squared_error(predictions, observations) ** 0.5
    
    def recordErrors(self, errorFileName, examples):
        sortedExamples = sorted(examples, key=lambda ex:self.rootMeanSquaredError([ex]), reverse=True)
        errorFile = open(errorFileName, 'w')
        errorFile.write("Example ID,RMS ERROR,User ID,Actual Response Time,Predicted Response Time,Question Length,Question Answer,Question\n")
        for ex in sortedExamples:
            raw_question_words = []
            for tuple in ex.question.question:
                raw_question_words.append(tuple[0])
            raw_question = " ".join(raw_question_words)
            raw_question = raw_question.replace(",","_")
            errorFile.write(",".join([str(ex.id), str(self.rootMeanSquaredError([ex])), str(ex.user), str(ex.observation), str(ex.prediction), str(len(raw_question.split())), ex.question.answer, str(raw_question)])+"\n")
        errorFile.flush()
        errorFile.close()
    
    def producePredictions(self, trainingExamples, testExamples, users, continuous_features, binary_features, wiki_data, category_dict, average):
        clusters = []
        if not self.cluster:
            clusters = [testExamples]
        else:
            answer_ranges = [10,50,100]
            last_range = 0
            for num_answered_range in answer_ranges:
                current_cluster = []
                for testExample in testExamples:
                    if testExample.observation <= num_answered_range and testExample.observation > last_range:
                        current_cluster.append(testExample)
                clusters.append(current_cluster)
                last_range = num_answered_range 
                
        if 'kernel' in binary_features:
            kernel = binary_features['kernel']
        else:
            kernel = 'rbf'
            binary_features['kernel'] = 'rbf'
            
        if 'gamma' in binary_features:
            gamma = float(binary_features['gamma'])
        else:
            gamma = 0.0
            binary_features['gamma'] = str(gamma)
        for cluster in clusters:
            analyzer_abs = Analyzer(features=continuous_features)
            featurizer_abs = Featurizer(category_dict, features=continuous_features, use_default=False, analyzer=analyzer_abs)
            
            analyzer_sign = Analyzer(features=binary_features)
            featurizer_sign = Featurizer(category_dict, features=binary_features, use_default=False, analyzer=analyzer_sign)
        #     featurizer = Featurizer(use_default=True)
            y_train_abs = []
            y_train_sign = []
            for train_example in trainingExamples:
#                 y_train_abs.append(train_example.observation)
                y_train_abs.append(abs(train_example.observation))
                y_train_sign.append(sign(train_example.observation))
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
#             x_test_abs = featurizer_abs.test_feature(testExamples, users, wiki_data)
            x_test_abs = featurizer_abs.test_feature(cluster, users, wiki_data)
              
              
        #     classifier = SGDClassifier(loss='log', penalty='l2', shuffle=True)
        #     
        #     lr.fit(x_train, y_train)
          
        #     continuous_classifier = linear_model.LinearRegression()
        #     continuous_classifier = svm.NuSVR()
            continuous_classifier = ensemble.GradientBoostingRegressor(n_estimators=self.n_estimators)
        #     continuous_classifier = ensemble.AdaBoostRegressor(n_estimators=4000)
        #     continuous_classifier = ensemble.RandomForestRegressor()
        #     continuous_classifier = ensemble.BaggingRegressor()
              
            print "Fitting continuous classifier"  
            continuous_classifier.fit(x_train_abs.toarray(), y_train_abs)  
              
            print "Fit continuous training data"
        #     
            print "Predicting continuous classifier"
            predictions_abs = continuous_classifier.predict(x_test_abs.toarray())
        #     print predictions_abs
            
#             binary_classifier = SGDClassifier(loss='log', penalty='l2', shuffle=True)
            
#             binary_classifier = svm.SVC(kernel=kernel, class_weight='auto')
            binary_classifier = ensemble.GradientBoostingClassifier(n_estimators=1000)
#             binary_classifier = neighbors.KNeighborsClassifier()
            print "Fitting binary classifier"
            print "Generating binary text x"
            print "Generating binary training x"
            
            x_train_sign = featurizer_sign.train_feature(trainingExamples, users, wiki_data)
            
            x_test_sign = featurizer_sign.test_feature(cluster, users, wiki_data)
        #     
        #     
            binary_classifier = svm.SVC(kernel=kernel)
        #     print x_test_sign.toarray()[0]
        #     print x_train_sign.toarray()[0]
        # #     binary_classifier = neural_network.BernoulliRBM()
            binary_classifier.fit(x_train_sign.toarray(), y_train_sign)
        #     
            print "Fit binary classifier"
            print "Predicting binary classifier"
            predictions_binary = binary_classifier.predict(x_test_sign.toarray())
            
        #     print predictions_binary
            print "Features importance:"
            print continuous_classifier.feature_importances_
            
#             for i in range(len(predictions_abs)):
#                 current_prediction = int(ceil(predictions_abs[i]))
#                 current_question = cluster[i].question
#                 current_question_min = current_question.min
#                 if current_question.num_correct > current_question.num_incorrect:
#                     majority_sign = 1
#                 else:
#                     majority_sign = -1
#                 if abs(current_prediction) < current_question_min:
#                     current_prediction = sign(current_prediction)*current_question_min
#                 if sign(current_prediction) != majority_sign:
#                     current_prediction = majority_sign*abs(current_prediction) 
#                 cluster[i].prediction = current_prediction
#                 testExamples[i].prediction = int(ceil(predictions_abs[i]))
        
        for i in range(len(predictions_abs)):
            current_sign = predictions_binary[i]
            current_abs = predictions_abs[i]
#             cluster[i].prediction = int(ceil(random.choice([1, -1])*current_abs))
            cluster[i].prediction = int(ceil(current_sign*current_abs))
            
#             testExamples[i].prediction = int(ceil(current_sign*current_abs))
#             testExamples[i].prediction = int(ceil(current_sign*average))
        
    #     return predictions_abs, predictions_binary
        
        
    def recordCrossValidationResults(self, err, recordFile, features_abs_used, features_sign_used):
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
    
        
    def load_wiki_data(self, wiki_file):
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
       
    def store_wiki_features(self, wiki_file, wiki_data):
        outfile = open(wiki_file, 'wb')
        print "Storing Wikipedia data"
        pickle.dump(wiki_data, outfile)
        outfile.flush()
        outfile.close()

    def create_example_from_csv(self, users, row, questions, categories_dict, test=False):
        current_example = Example()
        current_example.id = row['id']
        q_id = int(row['question'])
        current_example.question = questions[q_id]
        current_example.user = int(row['user'])
        if 'position' in row:
            current_example.observation = int(float(row['position']))
        
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
            current_category = questions[q_id].category
            current_user.add_question(q_id, correct, position, answer, current_category)
            
            current_example.answer = answer
            current_example.question.add_question_occurence(position)
            
            
            categories_dict[current_category].add_occurrence(current_user, position)
            
        return current_example
    
    def create_examples(self, users, fileName, categories_dict, questions_dict, nTrainingSets=-1, generate_test=False, all_test=False, limit=-1):
        fin = open(fileName, 'rt')
        reader = csv.DictReader(fin)
        
        train_examples = []
        test_examples = []
        
        rowIndex = 0
        nTrainingSets = self.args.local_selection
        total = 0.0
        count = 0.0
        for row in reader:
            if limit > 0 and rowIndex >= limit:
                break
            if all_test or generate_test and nTrainingSets > 0 and rowIndex % nTrainingSets == 0:
                current_example = self.create_example_from_csv(users, row, questions_dict, categories_dict, test=True)
                test_examples.append(current_example)
            else:
                total += float(row['position'])
                count += 1
                current_example = self.create_example_from_csv(users, row, questions_dict, categories_dict)
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

         

        
    def run(self):
        usingWikipediaFeatures = self.position_features['wiki_answer'] or self.correctness_features['wiki_answer']
    
        if usingWikipediaFeatures and not self.args.regenerate_wiki:
            wiki_data_file = self.args.wiki_file
            wiki_data = self.load_wiki_data(wiki_data_file)
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
        
        trainingExamples, testExamples, average_abs = self.create_examples(users, trainingFile, categories, questions, nTrainingSets=self.args.local_selection, generate_test=True, limit=self.args.limit)
        
        
        print "\nRead data"
        
        if self.args.local:
            errors = []
                    
            self.producePredictions(trainingExamples, testExamples, users, self.position_features, self.correctness_features, wiki_data, categories, average_abs) 
            
            print "\nFinished prediction"
            
            err = self.rootMeanSquaredError(testExamples)
            
            self.recordCrossValidationResults(err, 'records.csv', self.position_features, self.correctness_features)
            
            self.recordErrors("errors.csv", testExamples)
            
            print "CROSS-VALIDATION RESULTS"
            print "ERROR: ", err#np.mean(errors)
    
        if self.args.predict:
            # Read test file
            print "\nWriting predictions file"
            testFile = "test.csv"
            
            testExamples = self.create_examples(users, testFile, categories, questions, all_test=True)
            
            trainingExamples = self.create_examples(users, trainingFile, categories, questions)
            
            # Generate predictions
            self.producePredictions(trainingExamples, testExamples, users, self.position_features, self.correctness_features, wiki_data, categories, average_abs)
         
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
            wiki_data_file = self.args.wiki_file
            if self.args.regenerate_wiki or len(wiki_data) > 0 and original_wiki_length != len(wiki_data):
                self.store_wiki_features(wiki_data_file, wiki_data)