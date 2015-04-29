import csv

from question import *
from example import *
from error import *
from predictor import *

if __name__ == "__main__":
    validation = True
    if validation:
        questionFile = "questions_validation.csv"
        trainingFile = "validation_train.csv"
        testFile = "validation_test.csv"
    else:
        questionFile = "questions_augmented.csv"
        trainingFile = "train.csv"
        testFile = "test.csv"

    # Load Question file into dict: question_id --> Question
    questions = loadQuestionDict(questionFile)

    # Load training and test sets
    trainingSet = loadExamples(trainingFile, questions)
    testSet = loadExamples(testFile, questions)

    # Create Predictions
    predictor = Predictor(questions)
    predictor.producePredictions(trainingSet, testSet)

    if validation:
        # Calculate cross-validation errors 
        err = Error(testSet)

        # Show round results
        print "RMSE:", err.RMSE
        print "   RMSE abs(time):", err.RMSE_time
        print "   ACCURACY:      ", err.accuracy
        err.confusion()
        print "-----------------------"

    # Produce submission file
    submissionFile = "submission.csv"
    fout = open(submissionFile, 'w')
    writer = csv.writer(fout)
    writer.writerow(("id","position"))
    for ex in testSet:
        writer.writerow((ex.id, ex.prediction))
    fout.close()


