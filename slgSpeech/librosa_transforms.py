#!/usr/bin/env python

# slegroux@ccrma.stanford.edu

import librosa
from IPython import embed
import torch
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


class MFCC(object):
    def __init__(self, sr, hop_length=512, n_mfcc=13):
        self.sr = sr
        self.hop_length = 512
        self.n_mfcc = 13

    def __call__(self, sample):
        # sample['X'] is a tensor
        audio, label = sample['X'], sample['y']
        audio = audio.numpy()
        #.reshape(audio.shape[1])
        mfcc = librosa.feature.mfcc(y=audio,
                                    sr=self.sr,
                                    hop_length=self.hop_length,
                                    n_mfcc=self.n_mfcc)
        mfcc = torch.from_numpy(mfcc).double()
        sample = { 'X': mfcc, 'y': label}
        return sample

    
class MelSpectrogram(object):
    def __init__(self, sr, hop_length=512, n_mels=128):
        self.sr = sr
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, sample):
        # sample['X'] is a tensor
        audio, label = sample['X'], sample['y']
        audio = audio.numpy()
        melspec = librosa.feature.melspectrogram(y=audio,
                                                 sr=self.sr,
                                                 hop_length=self.hop_length,
                                                 n_mels=self.n_mels)
        melspec = torch.from_numpy(melspec).double()
        sample = { 'X': melspec, 'y': label}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(sr={0}, hop_length={1}, n_mels={2})'.format(self.sr, self.hop_length, self.n_mels)

    
class Spectrogram(object):
    def __init__(self, sr, n_fft=2048, hop_length=None):
        self.sr = sr
        self.hop_length  = hop_length
        self.n_fft = n_fft

    def __call__(self, sample):
        # sample['X'] is a tensor
        audio, label = sample['X'], sample['y']
        audio = audio.numpy()
        specgram = librosa.stft(audio, self.n_fft, self.hop_length)
        specgram = torch.from_numpy(specgram).double()
        sample = { 'X': specgram, 'y': label}
        return sample

    
class LogPowerSpectrogram(object):
    def __init__(self, hop_length=None, n_fft=2048):
        self.hop_length = hop_length
        self.n_fft = n_fft

    def __call__(self, sample):
        # sample['X'] is a tensor
        audio, label = sample['X'], sample['y']
        audio = audio.numpy()
        #.reshape(audio.shape[1])
        logspec = librosa.core.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))**2)
        logspec = torch.from_numpy(logspec).double()
        sample = { 'X': logspec, 'y': label}
        return sample


class PadTrim(object):
    def __init__(self, max_len, fill_value=0.0):
        self.max_len = max_len
        self.fill_value = fill_value

    def __call__(self, sample):
        audio, label = sample['X'], sample['y']
        audio = audio.numpy()
        
        if self.max_len > len(audio):
            pad = np.ones((self.max_len - len(audio))) * self.fill_value
            audio = np.concatenate((audio, pad))
        elif self.max_len < len(audio):
            audio = audio[:self.max_len]
            
        sample = {'X': torch.from_numpy(audio).double(), 'y': label}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(max_len={0}, fill_value={1})'.format(self.max_len, self.fill_value)


class toPILtoTensor(object):
    def __init__(self):
        self.pil2t = ToTensor()

    def __call__(self, sample):
        spectrum, label = sample['X'], sample['y']
        spectrum = spectrum.numpy()
        im = Image.fromarray(spectrum)
        spectrum_tensor = self.pil2t(im)
        sample = {'X': spectrum_tensor, 'y': label}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(max_len={0}, fill_value={1})'.format(self.max_len, self.fill_value)



