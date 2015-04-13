import csv
import numpy as np
from csv import DictReader, DictWriter
import wikipedia

train = DictReader(open("train.csv", 'r'))

users = {}
userpercent = {}
questions = {}
questpercent = {}
qanswered = {}

for ii in train:
	users[int(ii['user'])]=[]
	userpercent[int(ii['user'])] = [0,0]
	questions[int(ii['question'])]=[]
	questpercent[int(ii['question'])] = [0,0]
	qanswered[int(ii['user'])] = 0


train = DictReader(open("train.csv", 'r'))

#put all positions for each user and each question in a dict
for ii in train:

	users[int(ii['user'])].append(int(ii['position']))
	# for user in users:
	# 	if int(ii['user'])==int(user):
	# 		users[user].append(int(ii['position'])))
			# print user, ii['position']


	if int(ii['position'])>0:
		userpercent[int(ii['user'])][0]+=1
		userpercent[int(ii['user'])][1]+=1
		questpercent[int(ii['question'])][0]+=1
		questpercent[int(ii['question'])][1]+=1
	else:
		userpercent[int(ii['user'])][1]+=1
		questpercent[int(ii['question'])][1]+=1



	questions[int(ii['question'])].append(int(ii['position']))
	# for quest in questions:
	# 	if int(ii['question'])==int(quest):
	# 		questions[quest].append(int(ii['position'])))

	qanswered[int(ii['user'])] += 1
	# for user in qanswered:
	# 	if int(ii['user'])==int(user):
	# 		qanswered[user] += 1

#get average value
for user in users:
	users[user] = round(np.mean(users[user]))

for quest in questions:
	questions[quest] = round(np.mean(questions[quest]))


with open('train.csv', 'rb') as train, open('train3.csv', 'wb') as train2:
	csvreader = csv.DictReader(train)
	fieldnames = csvreader.fieldnames + ['AvgUserPos'] + ['AvgQuestPos'] + \
										['RoundPos'] + ['QuestAnswered'] + \
										['UserPercent'] + ['QuestPercent']
	csvwriter = csv.DictWriter(train2, fieldnames)
	csvwriter.writeheader()
	for row in csvreader:
		row['AvgUserPos'] = users[int(row['user'])]
		row['AvgQuestPos'] = questions[int(row['question'])]
		row['RoundPos'] = round(int(row['position']),-1)
		row['QuestAnswered'] = qanswered[int(row['user'])]
		row['UserPercent'] = float(userpercent[int(row['user'])][0])/userpercent[int(row['user'])][1]
		row['QuestPercent'] = float(questpercent[int(row['question'])][0])/questpercent[int(row['question'])][1]

		# print row
		csvwriter.writerow(row)

