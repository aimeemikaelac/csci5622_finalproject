import csv

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
        # Augmented data
        self.mean_position = 0
        self.answer_accuracy = 0
        self.training_examples = 0

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
        try:
            question.mean_position = stringToFloat(row['mean position'])
            question.answer_accuracy = stringToFloat(row['answer accuracy'])
            question.training_examples = stringToInt(row['training examples'])
        except KeyError as e:
            pass
        # Add to dict
        qDict[question.id] = question
    fin.close()

    return qDict

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



