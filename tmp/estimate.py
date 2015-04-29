from random import randint
from random import random
import csv
import sys
import numpy as np
from sets import Set
from sklearn import linear_model

from question import *
from example import *
from error import *

def stringToInt(s):
    return int(float(s))

# -----------------------------------------------
# Predicting answer time
def absMeanPosition(trainingSet, testSet):
    # Calculate mean and standard deviation of training examples
    averageTime = np.mean([x.observed_time for x in trainingSet])

    # Produce predictions
    for example in testSet:
        example.predicted_time = averageTime

def meanPosition(trainingSet, testSet):
    # Calculate mean and standard deviation of training examples
    averageTime = np.mean([x.observation for x in trainingSet])

    # Produce predictions
    for example in testSet:
        example.prediction = averageTime


def meanQuestionTime(trainingSet, testSet):
    for example in testSet:
        example.predicted_time = example.question.mean_position


def features(ex):
    return [ex.user, len(ex.question.text), ex.question.id]

def predictLinearRegression(trainingSet, testSet):
    X = [ features(x) for x in trainingSet]
    Y = [x.observation for x in trainingSet]

    #clf = linear_model.LinearRegression()
    clf = linear_model.Ridge(alpha = .1)
    clf.fit(X,Y)
    print clf.coef_
    for example in testSet:
        X = features(example)
        Y = clf.predict(X)
        example.prediction = Y



# -----------------------------------------------
# Predicting answer accuracy
def allCorrect(trainingSet, testSet):
    for example in testSet:
        example.predicted_correctness = 1

def randomCorrect(trainingSet, testSet):
    percentCorrect = 85
    for example in testSet:
        x = random()
        if 100*x < percentCorrect:
            example.predicted_correctness = 1
        else:
            example.predicted_correctness = -1

def meanQuestionAccuracy(trainingSet, testSet):

    for example in testSet:
        percentCorrect = example.question.answer_accuracy
        if random() < percentCorrect:
            example.predicted_correctness = 1
        else:
            example.predicted_correctness = -1

# -----------------------------------------------

def producePredictions(trainingSet, testSet):
    # Predict answer time
    #averageTime = np.mean([x.observation for x in trainingSet])
    #for example in testSet:
    #    example.predicted_time = averageTime
    meanQuestionTime(trainingSet, testSet)

    # Predict answer accuracy 
    #allCorrect(trainingSet, testSet)
    meanQuestionAccuracy(trainingSet, testSet)

    # Combine predictions for final prediction 
    for example in testSet:
        example.prediction = example.predicted_correctness * example.predicted_time



# Read Question file into dict: question_id --> Question
questions = loadQuestionDict("questions_augmented.csv")

# Read training examples
examples = loadTrainingExamples("train.csv")
for ex in examples:
    ex.question = questions[ex.question]

# Define cross-validation parameters
nTrainingSets = 10
boundaryIndices = [int(x) for x in np.linspace(0, len(examples)-1, nTrainingSets+1)]
trainingSets = []
for i in range(len(boundaryIndices)-1):
    set_i = examples[boundaryIndices[i] : boundaryIndices[i+1]]
    trainingSets.append(set_i)

# Perform cross validation on training examples 
errors = []
for i in range(nTrainingSets):
    # Partition training examples into train and validation sets
    trainingExamples = []
    verificationExamples = trainingSets[i]
    for j in range(nTrainingSets):
        if j != i:
            trainingExamples = trainingExamples + trainingSets[j]

    # Generate predictions
    producePredictions(trainingExamples, verificationExamples)

    # Calculate errors 
    err = Error(verificationExamples)
    print err.RMSE, err.RMSE_time, err.accuracy
    errors.append(err)

print "CROSS-VALIDATION RESULTS"
RMSE = [x.RMSE for x in errors]
print "AVG ERROR: ", np.mean(RMSE)
print "min median max:   ", min(RMSE), "  ", np.median(RMSE), "  ", max(RMSE)
RMSE_time = [x.RMSE_time for x in errors]
print "AVG TIME ERROR: ", np.mean(RMSE_time)
print "min median max:   ", min(RMSE_time), "  ", np.median(RMSE_time), "  ", max(RMSE_time)
accuracy = [x.accuracy for x in errors]
print "AVG ACCURACY: ", np.mean(accuracy)
print "min median max:   ", min(accuracy), "  ", np.median(accuracy), "  ", max(accuracy)

# Read test file
testFile = "test.csv"
fin = open(testFile, 'rt')
reader = csv.DictReader(fin)
testExamples = []
for row in reader:
    example = Example()
    example.id = stringToInt(row['id'])
    example.question = questions[stringToInt(row['question'])]
    example.user = stringToInt(row['user'])
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


