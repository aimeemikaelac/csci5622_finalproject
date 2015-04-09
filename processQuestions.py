import csv
import numpy as np

from question import *
from example import *
from user import *

def stringToInt(s):
    return int(float(s))

def augmentQuestions(questions):
    print "GENERATING ADDITIONAL QUESTION DATA"
    # Get default values
    positions = [x.position for x in examples]
    abs_positions = [abs(x) for x in positions]
    accuracies = [((x/abs(x))+1)/2 for x in positions]
    global_mean_position = np.mean(abs_positions)
    global_answer_accuracy = np.mean(accuracies)
    print "MEAN POSITION:", global_mean_position
    print "AVG ACCURACY:", global_answer_accuracy

    # process questions
    for q in questions.values():
        # Get examples involving this question
        positions = [x.position for x in examples if x.question == q]
        q.training_examples = len(positions)
        if len(positions) > 0:
            abs_positions = [abs(x) for x in positions]
            accuracies = [((x/abs(x))+1)/2 for x in positions]
            q.mean_position = np.mean(abs_positions)
            q.answer_accuracy = np.mean(accuracies)
        else:
            q.mean_position = global_mean_position
            q.answer_accuracy = global_answer_accuracy

def generateUserData():
    userIDs = set([x.user for x in examples])
    users = []
    for id in userIDs:
        print "ID: ", id
        user = User()
        user.id = id
        user_examples = [x for x in examples if x.user == id]
        print len( user_examples)
    
    
        for cat in categories:
            cat_examples = [x for x in user_examples if x.question.category == cat]
            print "  CAT:", cat
            print "  ", len(cat_examples)


categories = ['Earth Science', 'Biology', 'Literature', 'Astronomy', 'Fine Arts', 'Other', 'Social Studies', 'Mathematics', 'Chemistry', 'Physics', 'History']

# Read Question file into dict: question_id --> Question
questions = loadQuestionDict("questions_augmented.csv")

# Read training examples
examples = loadTrainingExamples("train.csv")
for ex in examples:
    ex.question = questions[ex.question]

categories = set([x.category for x in questions.values()])
print categories

generateUserData()

#augmentQuestions(questions)
#writeQuestionFile(questions, "questions_augmented.csv")

