from random import randint
from random import random
from random import shuffle
import csv
import sys
import numpy as np
from sets import Set
from sklearn import linear_model
from sklearn.svm import SVC

from question import *
from example import *
from error import *
from user import *
from utility import *
from predictor import *

if __name__ == "__main__":
    # Load Question file into dict: question_id --> Question
    questions = loadQuestionDict("questions_augmented.csv")

    # Load training examples
    examples = loadTrainingExamples("train.csv")
    shuffle(examples)
    for ex in examples:
        ex.question = questions[ex.question]

    #examples = examples[:int(.1 * len(examples))]
    # Initialize Predictor
    predictor = Predictor(questions)

    # Cross validation 
    training_set_size = .1
    boundary = int(training_set_size * len(examples))
    validation_test_set = examples[:boundary ]
    validation_training_set = examples[boundary:]

    predictor.producePredictions(validation_training_set, validation_test_set)

    # Calculate cross-validation errors 
    err = Error(validation_test_set)

    # Show round results
    print "RMSE:", err.RMSE
    print "   RMSE abs(time):", err.RMSE_time
    print "   ACCURACY:      ", err.accuracy
    err.confusion()
    print "-----------------------"

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
    predictor.producePredictions(examples, testExamples)

    # Produce submission file
    submissionFile = "submission.csv"
    fout = open(submissionFile, 'w')
    writer = csv.writer(fout)
    writer.writerow(("id","position"))
    for ex in testExamples:
        writer.writerow((ex.id, ex.prediction))
    fout.close()


