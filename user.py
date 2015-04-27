import csv
import numpy as np
from utility import *

class User:
    HEADER = ["id","training_examples"]
    CATEGORIES = ['Biology', 'Literature', 'Chemistry', 'Fine Arts', \
                'History', 'Other', 'Social Studies', 'Mathematics', \
                'Astronomy', 'Physics', 'Earth Science']
    def __init__(self, id):
        self.id = id

        self.examples = []
        self.mean_position = 0
        self.mean_accuracy = 0

        self.category_examples = {}
        self.category_position = {}
        self.category_accuracy = {}


def generateUserData(trainingSet, categories, questions):
    # Compiles additional data for each user based on training data.
    # - Average accuracy
    # - Average position 
    # - List of training examples
    # - Average accuracy for each category
    # - Average position for each category
    # - List of training examples for each category
    users = {}

    # User id -1 represents all training examples as if they were for a single user 
    user = User(-1)
    users[-1] = user
    users[-1].examples = trainingSet
    users[-1].mean_position = np.mean( [ abs(x.position) for x in trainingSet ]  )
    users[-1].mean_accuracy = np.mean( [ positionToAccuracy(x.position) for x in trainingSet ]  )
    for category in categories:
        categoryExamples = [x for x in trainingSet if x.question.category == category]
        users[-1].category_examples[category] = categoryExamples
        users[-1].category_position[category] = np.mean( [ abs(x.position) for x in categoryExamples ] )
        users[-1].category_accuracy[category] = np.mean( [ positionToAccuracy(x.position) for x in categoryExamples] )

    # Training data for specific users 
    user_IDs = set([x.user for x in trainingSet])
    for id in user_IDs:
        user = User(id)
        users[id] = user
        # List of training examples for this user
        user.examples = [x for x in trainingSet if x.user == user.id]
        # Average position for all user's training examples
        user.mean_position = np.mean([ abs(x.position)  for x in user.examples])
        # Average accuracy for all user's training examples
        user.mean_accuracy = np.mean([ positionToAccuracy(x.position)  for x in user.examples])
        # Per-category data
        for category in categories:
            # List of training examples for this category
            examples = [x for x in users[-1].category_examples[category] if x.user == user.id]
            user.category_examples[category] = examples
            if len(examples) > 0:
                user.category_position[category] = np.mean([ abs(x.position) for x in examples])
                user.category_accuracy[category] = np.mean([ positionToAccuracy(x.position) for x in examples])
            else:
                user.category_position[category] = users[-1].category_position[category]
                user.category_accuracy[category] = users[-1].category_accuracy[category]

    return users
