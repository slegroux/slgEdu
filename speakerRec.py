import numpy as np
import os
import pickle
import librosa
from vad import remove_silence

from gmms import GMMS, GMM

class speakerID(object):

    def __init__(self):
        self.features = []
        self.gmmset = GMMS()
        self.classes = []

    def enroll(self, name, signal, fs = 44100):
        signal_new = remove_silence(fs, signal)
        hop_length = np.min([0.016 * fs, 512])
        mfcc = librosa.feature.mfcc(y = signal_new, sr = fs, n_mfcc = 13, hop_length = hop_length)
        mfcc = mfcc.T
        mu = np.mean(mfcc, axis = 0)
        sigma = np.std(mfcc, axis = 0)
        feature = (mfcc - mu) / sigma
        self.features.append(feature)
        self.classes.append(name)

    def _get_gmm_set(self):
        return GMMS()

    def train(self):
        self.gmmset = self._get_gmm_set()
        for name, feats in zip(self.classes, self.features):
            self.gmmset.fit_new(feats, name)

    def predict(self, signal, fs = 44100):
        signal_new = remove_silence(fs, signal)
        hop_length = np.min([0.016 * fs, 512])
        mfcc = librosa.feature.mfcc(y = signal_new, sr = fs, n_mfcc = 13, hop_length = hop_length)
        mfcc = mfcc.T
        mu = np.mean(mfcc, axis = 0)
        sigma = np.std(mfcc, axis = 0)
        feature = (mfcc - mu) / sigma
        return self.gmmset.predict_one(feature)

    def recognize(self, name, signal, step = 1, duration = 1.5, fs = 44100, disp = True):
        totalChunks = 0
        totalCorrect = 0
        head = 0
        totallen = np.round(signal.shape[0] / fs).astype(int)
        predictions = []
        while head < totallen:
            totalChunks += 1
            tail = head + duration
            if tail > totallen:
                tail = totallen
            signali = signal[fs * head : np.min([fs * tail, fs * totallen])]
            predicted = self.predict(signal, fs)
            predictions.append(predicted)
            if predicted == name:
                totalCorrect += 1
            head += step

        print 'For', name, 'accuracy:', float(totalCorrect) / totalChunks

    def dump(self, fname, part = None):
        with open(fname, 'wb') as f:
            if part is None:
                pickle.dump(self, f, -1)
            else:
                pickle.dump(part, f, -1)

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as f:
            R = pickle.load(f)
            return R
