from os import listdir
from os.path import isfile, join, expanduser
from scipy.io.wavfile import read

from speakerID import *

DATASET = 'berkeley'
SPEAKERS = ['antonio', 'leonard']

TRAIN_TIME_S = 300
TEST_TIME_S = 10000

def concatFilesInDir(dir):
	files = [ f for f in listdir(dir) if isfile(join(dir, f)) ]
	allSignal = []
	fs = 0
	for file in files:
		if 'clean' in file:
			fs, signal = read(dir+'/'+file)
			if len(signal.shape) > 1:
				allSignal.extend(signal[:,0])
			else:
				allSignal.extend(signal)
	return np.array(allSignal), fs


def enroll(model):
	for speaker in SPEAKERS:
		dir = '{0}/wav/{1}/train/{2}'.format(expanduser('~'), DATASET, speaker)
		signal, fs = concatFilesInDir(dir)
		if TRAIN_TIME_S is None:
			model.enroll(speaker, signal, fs)
		else:
			model.enroll(speaker, signal[0:fs*TRAIN_TIME_S], fs)


def test(model):
	for speaker in SPEAKERS:
		dir = '{0}/wav/{1}/test/{2}'.format(expanduser('~'), DATASET, speaker)
		signal, fs = concatFilesInDir(dir)
		if TEST_TIME_S is None:
			model.recognize(speaker, signal, step=1, duration=1, fs=fs)
		else:
			model.recognize(speaker, signal[0:fs*TEST_TIME_S], step=0.3, duration=0.3, fs=fs)

if __name__ == "__main__":
	model = speakerID()
	enroll(model)
	model.train()
	test(model)
