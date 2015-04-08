import csv
import numpy as np
from csv import DictReader, DictWriter

train = DictReader(open("train.csv", 'r'))

users = {}
questions = {}

for ii in train:
	users[int(ii['user'])]=[]
	questions[int(ii['question'])]=[]


train = DictReader(open("train.csv", 'r'))

#put all positions for each user and each question in a dict
for ii in train:
	for user in users:
		if int(ii['user'])==int(user):
			users[user].append(abs(int(ii['position'])))
			# print user, ii['position']

	for quest in questions:
		if int(ii['question'])==int(quest):
			questions[quest].append(abs(int(ii['position'])))

#get average value
for user in users:
	users[user] = round(np.mean(users[user]))

for quest in questions:
	questions[quest] = round(np.mean(questions[quest]))


with open('train.csv', 'rb') as train, open('train2.csv', 'wb') as train2:
	csvreader = csv.DictReader(train)
	fieldnames = csvreader.fieldnames + ['AvgUserPos'] + ['AvgQuestPos'] + \
										['RoundPos']
	csvwriter = csv.DictWriter(train2, fieldnames)
	csvwriter.writeheader()
	for row in csvreader:
		row['AvgUserPos'] = users[int(row['user'])]
		row['AvgQuestPos'] = questions[int(row['question'])]
		row['RoundPos'] = round(int(row['position']),-1)
		# print row
		csvwriter.writerow(row)

