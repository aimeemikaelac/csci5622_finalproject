import csv
import numpy as np
from utility import *

def stringToInt(s):
    return int(float(s))

def stringToFloat(s):
    return float(s)

class Question:
    HEADER = ["question","answer","unknown","category","text","words","mean position","answer accuracy","training examples"]
    def __init__(self):
        # Data read from original questions.csv
        self.id = 0
        self.answer = ""
        self.unknown = ""  #don't know what this column means
        self.category = ""
        self.text = ""
        self.words = ""
        # Augmented data - derived from trainingSet in generateQuestionData()
        self.mean_position = -1
        self.answer_accuracy = -1
        self.examples = []

def loadQuestionDict(questionFile):
    # Read Question file into dict: question_id --> Question
    fin = open(questionFile, 'rt')
    reader = csv.DictReader(fin)
    qDict = {}
    for row in reader:
        question = Question()
        # Data found in original questions.csv
        question.id = stringToInt(row['question'])
        question.answer = row['answer']
        question.unknown = row['unknown']
        question.category = row['category']
        question.text = row['text']
        question.words = row['words']
        # Augmented data
        #try:
        #    question.mean_position = stringToFloat(row['mean position'])
        #    question.answer_accuracy = stringToFloat(row['answer accuracy'])
        #    question.training_examples = stringToInt(row['training examples'])
        #except KeyError as e:
        #    pass

        # Add to dict
        qDict[question.id] = question
    fin.close()

    return qDict

def generateQuestionData(questions, trainingSet):
    # Synthesizes additional data for each question based on training data.
    # - Average position per question from training data
    # - Average accuracy per question from training data

    # Compute default
    positions = [x.position for x in trainingSet]
    abs_positions = [abs(x) for x in positions]
    accuracies = [  positionToAccuracy(x)  for x in positions] # -1 or 1
    global_mean_position = np.mean(abs_positions)
    global_answer_accuracy = np.mean(accuracies)

    # process questions
    for q in questions.values():
        # Extract training examples for this question
        q.examples = [x for x in trainingSet if x.question == q]

        if len(q.examples) > 0:
            positions = [x.position for x in q.examples]
            # Compute average position
            abs_positions = [abs(x) for x in positions]
            q.mean_position = np.mean(abs_positions)
            # Compute average accuracy
            accuracies = [ positionToAccuracy(x)  for x in positions] # 0 or 1
            q.answer_accuracy = np.mean(accuracies)
        else:
            # This questions was not in training data. Use average over all questions
            q.mean_position = global_mean_position
            q.answer_accuracy = global_answer_accuracy


def writeQuestionFile(questions, augmentedFile):
    fout = open(augmentedFile, 'w')
    writer = csv.writer(fout)
    writer.writerow(Question.HEADER)
    for q in questions.values():
        writer.writerow((q.id,
                        q.answer,
                        q.unknown,
                        q.category,
                        q.text,
                        q.words,
                        q.mean_position,
                        q.answer_accuracy,
                        q.training_examples))
    fout.close()



