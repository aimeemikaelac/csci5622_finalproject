import csv

from question import *


def stringToInt(s):
    return int(float(s))

class Example:
    def __init__(self):
        # Training data 
        self.id = -1
        self.question_id = -1
        self.user_id = -1
        self.position = 0
        self.answer = ""
        # Question object
        self.question = None
        # Observed position from training data
        self.observation =  0           # 'position' from training data. eg. -78, 67
        self.observed_time = 0          # a positive number
        self.observed_correctness = 0   # -1 or 1
        # Predicted position
        self.prediction = 0             # Final prediction. eg. -78, 67
        self.predicted_time = 0         # a positive number
        self.predicted_correctness = 0  # -1 or 1

def loadExamples(exampleFile, questions):
    # Read examples from training or test file
    fin = open(exampleFile, 'rt')
    reader = csv.DictReader(fin)
    examples = []
    for row in reader:
        example = Example()
        # Values in all training and test sets 
        example.id = row['id']
        example.question_id = row['question']
        example.user_id = row['user']
        example.question = questions[example.question_id]

        # Values that are not in final test set
        try:
            example.position = stringToInt(row['position'])
            example.answer = row['answer']
            example.observation = stringToInt(row['position'])
            example.observed_time = abs(example.observation)
            example.observed_correctness = example.observation / example.observed_time
        except KeyError:
            pass
 
        examples.append(example)
    fin.close()

    return examples

