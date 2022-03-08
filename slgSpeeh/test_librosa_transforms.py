#!/usr/bin/env python

import unittest
import librosa
from librosa_transforms import MFCC
from FSSDData import FSDDDataFromDir, create_partition
from IPython import embed
import numpy as np
import torchaudio
import torch
from torch.utils import data
from matplotlib import pyplot as plt

DATA_DIR = '/home/workfit/Sylvain/Data/free-spoken-digit-dataset/recordings'

PARAMS= {
    'batch_size': 1,
    'shuffle': True,
    'num_workers': 1
    }

class MFCCTest(unittest.TestCase):
    def setUp(self):
        self.fn = '/home/workfit/Sylvain/Projects/SpeakerID/Tests/7_nicolas_34.wav'
        X, self.sr = librosa.load(self.fn)
        self.X = torch.from_numpy(X)
        self.y = ['seven']
        self.sample = {'X': self.X, 'y': self.y}
        self.mfcc = MFCC(self.sr)

    def tearDown(self):
        del self.sample
        del self.sr
        del self.mfcc

    def test_sr(self):
        self.assertEqual(self.sr, 22050)

    def test_X(self):
        print(type(self.X))

    def test_torchaudio(self):
        X, sr = torchaudio.load(self.fn)
        
    def test_length_audio(self):
        self.assertEqual(len(self.sample['X']), 6726)

    def test_transform_sample(self):
        transformed = self.mfcc(self.sample)
        self.assertEqual(transformed['X'].shape, (13,14))

    def test_create_partition(self):
        # len validation = 20 % 1500 utterances
        partition, labels, wav_scp = create_partition(DATA_DIR)
        self.assertEqual(np.shape(partition['validation']), (300,))

    def test_create_dataset_from_dir(self):
        partition, labels, wav_scp = create_partition(DATA_DIR)
        validation_ds = FSDDDataFromDir(partition['validation'], labels, wav_scp)
        self.assertEqual(len(validation_ds.utt_ids), 300)

    def test_create_dataloader_from_dir(self):
        partition, labels, wav_scp = create_partition(DATA_DIR)
        validation_ds = FSDDDataFromDir(partition['validation'], labels, wav_scp)
        validation_dl = data.DataLoader(validation_ds, **PARAMS)
        dataiter = iter(validation_dl)
        test_sample = dataiter.next()
        self.assertEqual(len(validation_ds.utt_ids), 300)

    def test_create_dataloader_with_tsfm(self):
        partition, labels, wav_scp = create_partition(DATA_DIR)
        validation_ds = FSDDDataFromDir(partition['validation'], labels, wav_scp, transform=self.mfcc)
        validation_dl = data.DataLoader(validation_ds, **PARAMS)
        dataiter = iter(validation_dl)
        test_sample = dataiter.next()
        self.assertEqual(len(validation_ds.utt_ids), 300)

    def test_tsfm(self):
        pass

if __name__ == '__main__':
    unittest.main()
