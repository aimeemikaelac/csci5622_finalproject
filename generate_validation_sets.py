
from random import shuffle
import csv
from question import *
from example import *

# Load questions
questionInputFile = "questions_augmented.csv"
questions = loadQuestionDict(questionInputFile)

# Load training examples
fullTrainingSet = "train.csv"
examples = loadExamples(fullTrainingSet, questions)
shuffle(examples)

# Partition validation sets
training_set_size = .1
boundary = int(training_set_size * len(examples))
validation_test_set = examples[:boundary ]
validation_training_set = examples[boundary:]

# Save validation training set
validation_train_file = "validation_train.csv"
fout = open(validation_train_file, 'w')
writer = csv.writer(fout)
writer.writerow(("id","question","user","position","answer"))
for ex in validation_training_set:
    writer.writerow((str(ex.id), str(ex.question_id), str(ex.user_id),ex.observation, ex.answer))
fout.close()

# Save validation test set
validation_test_file = "validation_test.csv"
fout = open(validation_test_file, 'w')
writer = csv.writer(fout)
writer.writerow(("id","question","user","position","answer"))
for ex in validation_test_set:
    writer.writerow((str(ex.id), str(ex.question_id), str(ex.user_id),ex.observation, ex.answer))
fout.close()

# Generate augmented question data
augmentedQuestionFile = "questions_validation.csv"
generateQuestionData(questions, validation_training_set)
writeQuestionFile(questions, augmentedQuestionFile)
