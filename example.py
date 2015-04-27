import csv

from question import *


def stringToInt(s):
    return int(float(s))

class Example:
    def __init__(self):
        # Training data 
        self.id = -1
        self.question = -1
        self.user = -1
        self.position = 0
        self.answer = ""
        # Observed position from training data
        self.observation =  0           # 'position' from training data. eg. -78, 67
        self.observed_time = 0          # a positive number
        self.observed_correctness = 0   # -1 or 1
        # Predicted position
        self.prediction = 0             # Final prediction. eg. -78, 67
        self.predicted_time = 0         # a positive number
        self.predicted_correctness = 0  # -1 or 1

def loadTrainingExamples(trainingFile):
    # Read training examples
    fin = open(trainingFile, 'rt')
    reader = csv.DictReader(fin)
    examples = []
    for row in reader:
        example = Example()
        example.id = stringToInt(row['id'])
        example.question = stringToInt(row['question'])
        example.user = stringToInt(row['user'])
        example.position = stringToInt(row['position'])
        example.answer = row['answer']
        #example.question = questions[stringToInt(row['question'])]
        example.observation = stringToInt(row['position'])
        example.observed_time = abs(example.observation)
        example.observed_correctness = example.observation / example.observed_time
        examples.append(example)
    fin.close()
    return examples


