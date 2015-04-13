#stats

from collections import defaultdict
import csv
from csv import DictReader, DictWriter


d = defaultdict(int)
bins = defaultdict(int)

train = DictReader(open("train2.csv", 'r'))

for ii in train:
	d[int(ii['user'])]+=1

for user in d:
	# print user, d[user]
	if d[user]==1:
		bins[1]+=1
	elif d[user]<=5:
		bins[5]+=1
	elif d[user]<=15:
		bins[15]+=1
	elif d[user]<=25:
		bins[25]+=1
	elif d[user]<=50:
		bins[50]+=1
	elif d[user]<=100:
		bins[100]+=1
	elif d[user]<=200:
		bins[200]+=1
	else:
		bins[201]+=1

for b in bins:
	print b, bins[b]
