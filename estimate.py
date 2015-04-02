
import csv
import sys
import numpy as np
from sets import Set
from sklearn.metrics import mean_squared_error

class Example:
    def __init__(self):
        self.id = 0
        self.question = ""
        self.user = 0
        self.answer = ""
        self.observation =  0
        self.prediction = 0

def stringToInt(s):
    return int(float(s))

def rootMeanSquaredError(examples):
    predictions = [x.prediction for x in examples]
    observations = [x.observation for x in examples]
    return mean_squared_error(predictions, observations) ** 0.5

def producePredictions(trainingSet, testSet):
    # Calculate mean and standard deviation of training examples
    averageTime = np.mean([x.observation for x in trainingSet])
    stdDev = np.std([x.observation for x in trainingSet])

    # Produce predictions
    for example in testSet:
        example.prediction = averageTime

if __name__ == "__main__":
    # Read training examples
    trainingFile = "train.csv"
    fin = open(trainingFile, 'rt')
    reader = csv.DictReader(fin)
    examples = []
    for row in reader:
        example = Example()
        example.id = row['id']
        example.question = row['question']
        example.user = row['user']
        example.observation = stringToInt(row['position'])
        example.answer = row['answer']
        examples.append(example)
    fin.close()

    #    example.observation = (example.observation - mean)/dev

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

        # Calculate root-mean-square deviation
        err = rootMeanSquaredError(verificationExamples)
        errors.append(err)

    print "CROSS-VALIDATION RESULTS"
    print "AVG ERROR: ", np.mean(errors)
    print "min median max:   ", min(errors), "  ", np.median(errors), "  ", max(errors)


    # Read test file
    testFile = "test.csv"
    fin = open(testFile, 'rt')
    reader = csv.DictReader(fin)
    testExamples = []
    for row in reader:
        example = Example()
        example.id = row['id']
        example.question = row['question']
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


