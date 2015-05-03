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
quest_numbers = []
user_cat_percent = {}
user_cat_pos = {}


for ii in train:
	users[int(ii['user'])]=[]
	userpercent[int(ii['user'])] = [0,0]
	questions[int(ii['question'])]=[]
	questpercent[int(ii['question'])] = [0,0]
	qanswered[int(ii['user'])] = 0
	quest_numbers.append(int(ii['question']))
	user_cat_percent[int(ii['user'])] = {}
	user_cat_pos[int(ii['user'])] = {}



cat_percent = {}
cat_pos = {}
question_data = {}
qq = DictReader(open("questions_processed.csv"),'r')
counter = 0
for ii in qq:
    # print ii
    # print ii['r']     #Question Number
    # print ii[None][0]     #Answer
    # print ii[None][2]       #Category
    # print ii[None][3]       #Question (all words)
    # print ii[None][4]       #Question (no stopwords)
    # print ii[None][5]       #Wiki article length (28000 avg, 0 was no article)
    if counter >0:
        question_data[int(float(ii['r']))] = [ii[None][0], ii[None][2], 
                                        ii[None][3], ii[None][4]]
        cat_percent[ii[None][2]] = [0,0]
        cat_pos[ii[None][2]] = []
    counter+=1
###question_data[id][0]=answer, [id][1]=caegory
### [id][2]=question and POS, [id][3]=question(no stopwords)


train = DictReader(open("train.csv", 'r'))
for ii in train:
	user_cat_percent[int(ii['user'])][question_data[int(ii['question'])][1]] = [0,0]
	user_cat_pos[int(ii['user'])][question_data[int(ii['question'])][1]] = []



train = DictReader(open("train.csv", 'r'))
#put all positions for each user and each question in a dict
for ii in train:

	users[int(ii['user'])].append(int(ii['position']))

	if int(ii['position'])>0:
		userpercent[int(ii['user'])][0]+=1
		userpercent[int(ii['user'])][1]+=1
		questpercent[int(ii['question'])][0]+=1
		questpercent[int(ii['question'])][1]+=1
		cat_percent[question_data[int(ii['question'])][1]][0]+=1
		cat_percent[question_data[int(ii['question'])][1]][1]+=1
		user_cat_percent[int(ii['user'])][question_data[int(ii['question'])][1]][0]+=1
		user_cat_percent[int(ii['user'])][question_data[int(ii['question'])][1]][1]+=1
	else:
		userpercent[int(ii['user'])][1]+=1
		questpercent[int(ii['question'])][1]+=1
		cat_percent[question_data[int(ii['question'])][1]][1]+=1
		user_cat_percent[int(ii['user'])][question_data[int(ii['question'])][1]][1]+=1

	questions[int(ii['question'])].append(int(ii['position']))
	qanswered[int(ii['user'])] += 1
	cat_pos[question_data[int(ii['question'])][1]].append(int(ii['position']))
	user_cat_pos[int(ii['user'])][question_data[int(ii['question'])][1]].append(int(ii['position']))

#get average values
avg_usr_pos = []
avg_usr_per = []
for user in users:
	users[user] = np.mean(users[user])
	avg_usr_pos.append(np.mean(users[user]))
	avg_usr_per.append(float(userpercent[user][0])/userpercent[user][1])
avg_usr_pos = np.mean(avg_usr_pos)
avg_usr_per = np.mean(avg_usr_per)


avg_quest_pos = []
avg_quest_per = []
for quest in questions:
	questions[quest] = np.mean(questions[quest])
	avg_quest_pos.append(np.mean(questions[quest]))
	avg_quest_per.append(float(questpercent[quest][0])/questpercent[quest][1])
avg_quest_pos = np.mean(avg_quest_pos)
avg_quest_per = np.mean(avg_quest_per)


for cat in cat_pos:
	cat_pos[cat] = np.mean(cat_pos[cat])
	# print cat, cat_pos[cat]

for user in user_cat_pos:
	for cat in user_cat_pos[user]:
		user_cat_pos[user][cat] = np.mean(user_cat_pos[user][cat])
		# print user_cat_pos[user], user_cat_pos[user][cat]

print avg_usr_pos
print avg_usr_per
print avg_quest_pos
print avg_quest_per


with open('train.csv', 'rb') as train, open('train3.csv', 'wb') as train2:
	csvreader = csv.DictReader(train)
	fieldnames = csvreader.fieldnames + ['AvgUserPos'] + ['AvgQuestPos'] + \
										['QuestAnswered'] + \
										['UserPercent'] + ['QuestPercent'] + \
										['CatPercent'] + ['UserCatPercent'] + \
										['CatPos'] + ['UserCatPos']
	csvwriter = csv.DictWriter(train2, fieldnames)
	csvwriter.writeheader()
	for row in csvreader:
		try:
			row['AvgUserPos'] = users[int(row['user'])]
		except KeyError:
			row['AvgUserPos'] = avg_usr_pos
		try:
			row['AvgQuestPos'] = questions[int(row['question'])]
		except KeyError:
			row['AvgQuestPos'] = avg_quest_pos
		try:
			row['QuestAnswered'] = qanswered[int(row['user'])]
		except KeyError:
			row['QuestAnswered'] = 1
		try:
			row['UserPercent'] = float(userpercent[int(row['user'])][0])/userpercent[int(row['user'])][1]
		except KeyError:
			row['UserPercent'] = avg_usr_per
		try:
			row['QuestPercent'] = float(questpercent[int(row['question'])][0])/questpercent[int(row['question'])][1]
		except KeyError:
			row['QuestPercent'] = avg_quest_per
		try:
			row['CatPercent'] = (float(cat_percent[question_data[int(row['question'])][1]][0])/
									float(cat_percent[question_data[int(row['question'])][1]][1]))
		except KeyError:
			row['CatPercent'] = avg_quest_per
		try:
			row['UserCatPercent'] = (float(user_cat_percent[int(row['user'])][question_data[int(row['question'])][1]][0])/
										float(user_cat_percent[int(row['user'])][question_data[int(row['question'])][1]][1]))
		except KeyError:
			row['UserCatPercent'] = avg_usr_per
		try:
			row['UserCatPos'] = float(user_cat_pos[int(row['user'])][question_data[int(row['question'])][1]])
										
		except KeyError:
			row['UserCatPos'] = avg_usr_pos
		try:
			row['CatPos'] = float(cat_pos[question_data[int(row['question'])][1]])
		except:
			row['CatPos'] = avg_quest_pos


		# print row
		csvwriter.writerow(row)

