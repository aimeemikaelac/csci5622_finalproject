
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from question import *
from user import *

class Predictor:
    def __init__(self, questions):
        self.questions = questions
        self.categories = set([x.category for x in questions.values()])
        self.users = {}

        #self.positionPredictor = Position_Regression()
        self.positionPredictor = Position_Mean()

        #self.accuracyPredictor = Accuracy_Regression()
        self.accuracyPredictor = Accuracy_AllCorrect()

    def producePredictions(self, trainingSet, testSet):
        # Generate some additional data from training set
        print "GENERATE ADDITIONAL QUESTION DATA"
        generateQuestionData(self.questions, trainingSet)
        print "GENERATE USER DATA"
        self.users = generateUserData(trainingSet, self.categories, self.questions)
        print "CALCULATE PREDICTIONS"
        # Predict time-to-answer
        self.positionPredictor.users = self.users
        self.positionPredictor.predict(trainingSet, testSet)
        #meanPosition(trainingSet, testSet)

        # Predict answer accuracy 
        #allCorrect(trainingSet, testSet)
        self.accuracyPredictor.users = self.users
        self.accuracyPredictor.predict(trainingSet, testSet)
        #regressionAccuracy(trainingSet, testSet)

        # Combine predictions for final prediction 
        for example in testSet:
            example.prediction = example.predicted_correctness * example.predicted_time


# ---------------------------------------------------
# class Position_Regression
# Predicts time-to-answer using Regression
# ---------------------------------------------------
class Position_Regression:
    def __init__(self):
        self.users = {}
        pass

    def predict(self, trainingSet, testSet):
        X = [self.features(x) for x in trainingSet]
        Y = [x.observed_time for x in trainingSet]
        logreg = linear_model.LinearRegression()
        logreg.fit(X,Y)
        for example in testSet:
            X = self.features(example)
            Y = logreg.predict(X)
            example.predicted_time = Y

    def features(self, ex):
        # Produce a vector of features for predicting time-to-answer
        try:
            user = self.users[ex.user]
        except KeyError:
            user = self.users[-1]
        features = [
                    # Question features
                    len(ex.question.text),
                    ex.question.answer_accuracy,
                    ex.question.mean_position,
                    len(ex.question.examples),
                    # User features
                    len(user.examples),
                    user.mean_accuracy,
                    user.mean_position,
                    len(user.category_examples[ex.question.category]),
                    user.category_accuracy[ex.question.category],
                    user.category_position[ex.question.category],
                    self.users[-1].category_accuracy[ex.question.category],
                    self.users[-1].category_position[ex.question.category],
                        ]
        return features

# ---------------------------------------------------
# class Accuracy_Regression
# Predicts answer accuracy using Logistic Regression
# ---------------------------------------------------
class Accuracy_Regression:
    def __init__(self):
        self.users = {}

    def predict(self, trainingSet, testSet):
        # Logistic regression to predict correctness
        X = [self.features(x) for x in trainingSet]
        Y = [x.observed_correctness for x in trainingSet]
        logreg = linear_model.LogisticRegression(C=10000)
        logreg.fit(X,Y)
        print logreg.coef_
        for example in testSet:
            X = self.features(example)
            Y = logreg.predict(X)
            example.predicted_correctness = Y[0]

    def features(self, ex):
        # Produce a vector of features for a specific example 
        try:
            user = self.users[ex.user]
        except KeyError:
            user = self.users[-1]
        features = [
                    # Question features
                    len(ex.question.text),
                    ex.question.answer_accuracy,
                    ex.question.mean_position,
                    len(ex.question.examples),
                    # User features
                    len(user.examples),
                    user.mean_accuracy,
                    user.mean_position,
                    len(user.category_examples[ex.question.category]),
                    user.category_accuracy[ex.question.category],
                    user.category_position[ex.question.category],
                    self.users[-1].category_accuracy[ex.question.category],
                    self.users[-1].category_position[ex.question.category],
                        ]
        return features

class Accuracy_AllCorrect:
    def __init__(self):
        self.users = {}

    def predict(self, trainingSet, testSet):
        for example in testSet:
            example.predicted_correctness = 1


class Position_AbsoluteMean:
    def __init__(self):
        self.users = {}

    def predict(self, trainingSet, testSet):
        averageTime = np.mean([abs(x.observation) for x in trainingSet])
        for example in testSet:
            example.predicted_time = averageTime

class Position_Mean:
    def __init__(self):
        self.users = {}

    def predict(self, trainingSet, testSet):
        averageTime = np.mean([x.observation for x in trainingSet])
        for example in testSet:
            example.predicted_time = averageTime

# -----------------------------------------------
# Predicting answer accuracy
#


class Accuracy_SVM:
    def __init__(self):
        self.users = {}

    def predict(self, trainingSet, testSet):
        # SVM to predict correctness
        X = [accuracyFeatures(x) for x in trainingSet]
        Y = [x.observed_correctness for x in trainingSet]
        svc = SVC()
        svc.fit(X,Y)
        for example in testSet:
            X = self.features(example)
            Y = svc.predict(X)
            example.predicted_correctness = Y[0]

    def features(self, ex):
        # Produce a vector of features for a specific example 
        try:
            user = self.users[ex.user]
        except KeyError:
            user = self.users[-1]
        features = [
                    # Question features
                    len(ex.question.text),
                    ex.question.answer_accuracy,
                    ex.question.mean_position,
                    len(ex.question.examples),
                    # User features
                    len(user.examples),
                    user.mean_accuracy,
                    user.mean_position,
                    len(user.category_examples[ex.question.category]),
                    user.category_accuracy[ex.question.category],
                    user.category_position[ex.question.category],
                    self.users[-1].category_accuracy[ex.question.category],
                    self.users[-1].category_position[ex.question.category],
                        ]
        return features




