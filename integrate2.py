import csv
import numpy as np
from csv import DictReader, DictWriter
import wikipedia

wiki = {}
question_data = {}
questions = DictReader(open("questions.csv"),'r')
for ii in questions:
    # print ii['r']     #Question Number
    # print ii[None][0]     #Answer
    # print ii[None][2]       #Category
    # print ii[None][3]       #Question (all words)
    # print ii[None][4]       #Question (no stopwords)

    # print ''
    question_data[int(ii['r'])] = ii[None][0]

for qq in question_data:
	# print qq, question_data[qq]

	try:
		article = wikipedia.page(question_data[qq])
		wiki[qq] = len(article.content)
	except wikipedia.exceptions.DisambiguationError:
		wiki[qq] = 28000 
	except wikipedia.exceptions.PageError:
		wiki[qq] = 0


row = {}
# Write predictions
o = DictWriter(open('wiki.csv', 'wb'), ['question', 'WikiLength'])
o.writeheader()
for ii in wiki:
	row['question'] = ii
	row['WikiLength'] = wiki[ii]
	o.writerow(row)

