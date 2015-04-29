import csv
import numpy as np
from utility import *
from textstat.textstat import textstat

def stringToInt(s):
    return int(float(s))

def stringToFloat(s):
    return float(s)

class Question:
    HEADER = ["question","answer","unknown","category","text","words","mean position","answer accuracy","training examples",
            "syllables","sentences","lexicon","grade","ease",
            "automated","smog","fog","coleman","linsear","dale","consensus","difficult"]
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
        self.nPeriods = -1
        # Lexical data
        self.syllables = -1
        self.sentences = -1
        self.grade_level = -1
        self.reading_ease = -1
        self.lexicon = -1
        
        self.automated_readability = -1
        self.smog = -1
        self.fog = -1
        self.coleman_liau = -1
        self.linsear_write = -1
        self.dale_chall = -1
        self.readability_consensus = -1
        self.difficult = -1


def loadQuestionDict(questionFile):
    # Read Question file into dict: question_id --> Question
    fin = open(questionFile, 'rt')
    reader = csv.DictReader(fin)
    qDict = {}
    for row in reader:
        question = Question()
        # Data found in original questions.csv
        question.id = row['question']
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
 
            question.syllables = stringToFloat(row['syllables'])
            question.sentences = stringToFloat(row['sentences'])
            question.lexicon = stringToFloat(row['lexicon'])
            question.grade_level = stringToFloat(row['grade'])
            question.reading_ease = stringToFloat(row['ease'])

            question.automated_readability = stringToFloat(row['automated'])
            #question.smog = stringToFloat(row['smog'])
            question.fog = stringToFloat(row['fog'])
            question.coleman_liau = stringToFloat(row['colemant'])
            question.linsear_write = stringToFloat(row['linsear'])
            question.dale_chall = stringToFloat(row['dale'])
            #question.readability_consensus = stringToFloat(row['consensus'])
            question.difficult = stringToFloat(row['difficult'])


        except KeyError as e:
            pass
        except ValueError as e:
            print e
            print row['smog']
            raise

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

        # Lexigraphic data
        q.syllables = textstat.syllable_count(q.text)
        q.sentences = textstat.sentence_count(q.text)
        q.grade_level = textstat.flesch_kincaid_grade(q.text)
        q.reading_ease = textstat.flesch_reading_ease(q.text)
        q.lexicon = textstat.lexicon_count(q.text)
        q.automated_readability = textstat.automated_readability_index(q.text) 
        q.coleman_liau = textstat.coleman_liau_index(q.text)
        # Some problem cases
        try:
            q.smog = textstat.smog_index(q.text)
            q.fog = textstat.gunning_fog(q.text)
            q.linsear_write = textstat.linsear_write_formula(q.text)
            q.dale_chall = textstat.dale_chall_readability_score(q.text)
            q.readability_consensus = textstat.readability_consensus(q.text)
            q.difficult = textstat.difficult_words(q.text)
        except IndexError as e:
            print e
            print q.text
            pass
        except TypeError as e:
            print e
            print q.text
            pass

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

    avg_linsear = np.mean([x.linsear_write for x in questions.values()])
    avg_smog = np.mean([x.smog for x in questions.values()])
    avg_fog = np.mean([x.fog for x in questions.values()])
    avg_difficult = np.mean([x.difficult for x in questions.values()])
    avg_dale = np.mean([x.dale_chall for x in questions.values()])

    for q in questions.values():
        if q.smog == -1 or q.smog == "":
            q.smog = avg_smog
        if q.fog == -1 or q.fog == "":
            q.fog = avg_fog
        if q.linsear == -1 or q.linsear == "":
            q.linsear = avg_linsear
        if q.dale == -1 or q.dale == "":
            q.dale = avg_dale
        if q.difficult == -1 or q.difficult == "":
            q.difficult = avg_difficult


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
                        len(q.examples),
                        q.syllables,
                        q.sentences,
                        q.lexicon,
                        q.grade_level,
                        q.reading_ease,

                        q.automated_readability, 
                        q.smog, 
                        q.fog,
                        q.coleman_liau, 
                        q.linsear_write,
                        q.dale_chall,
                        q.readability_consensus ,
                        q.difficult



                        ))


    fout.close()



