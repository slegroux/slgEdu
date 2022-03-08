#!/usr/bin/env python
import os
from sklearn.cross_validation import train_test_split
import pickle

train = {}
test = {}

for folder in ['DC', 'JE', 'JK', 'KL']:
	file_names = os.listdir(folder)
	train[folder], test[folder] = train_test_split(file_names, test_size=0.1)

pickle.dump( [train, test], open( "data.pkl", "wb" ) )
