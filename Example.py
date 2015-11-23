import csv

from User import User


class Example:
    def __init__(self):
        self.id = 0
        self.question = None
        self.user = 0
        self.answer = ""
        self.previous_prediction = 0.0
        #observed value of response time from training data
        self.observation =  0
        #predicted value obtained from classifier
        self.prediction = 0

#     def __srt__(self):
#         return str(self.id) +","+