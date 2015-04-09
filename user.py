import csv

class User:
    HEADER = ["id","training_examples"]
    def __init__(self):
        self.userID = -1

        self.mean_position = 0
        self.answer_accuracy = 0
        self.training_examples = 0


