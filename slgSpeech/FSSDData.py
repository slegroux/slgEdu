#!/usr/bin/env python

import glob, os
import json
import re
import random
from IPython import embed
import pdb
import torch
from torch.utils import data
import librosa

DATA_DIR = '/home/workfit/Sylvain/Data/free-spoken-digit-dataset/recordings'
CURRENT_DIR = os.getcwd()


def create_partition_from_dir(data_dir):

    digit2text = {
        '0': "zero",
        '1':"one",
        '2':"two",
        '3':"three",
        '4':"four",
        '5':"five",
        '6':"six",
        '7':"seven",
        '8':"eight",
        '9':"nine"
    }
    speakers = {'jackson', 'nicolas', 'theo'}

    expression = r"_[0-9].wav"
    speaker_regex = r"_(.*)_"
    digit_regex = r"^\d+"

    wav_scp = {} 
    labels = {}
    utt2spk = {}
    partition = {}
 
    os.chdir(data_dir)
    counter = 0
    for file in glob.glob("*.wav"):
        utt_id = os.path.splitext(file)[0]
        wav_path = os.path.join(data_dir, file)    
        wav_scp[utt_id] = wav_path
        digit = re.match(digit_regex, file).group()
        labels[utt_id] = digit2text[digit]
        speaker = re.search(speaker_regex, file).group(1)
        utt2spk[utt_id] = speaker
        counter += 1
    print('number of utterances: ', counter)
    spk2utt = {}
    for utt, spk in utt2spk.items():
        if spk not in spk2utt:
            spk2utt[spk] = [utt]
        else:
            spk2utt[spk].append(utt)

    partition = {'validation': [], 'train': []}
    for speaker in speakers:
        utts = spk2utt[speaker]
        utts.sort()
        random.seed(230)
        random.shuffle(utts) 
        partition['validation'] += utts[:100]
        partition['train'] += utts[100:]
    partition['validation'].sort()
    random.seed(230)
    random.shuffle(partition['validation'])
    partition['train'].sort()
    random.seed(230)
    random.shuffle(partition['train'])
    
    
    os.chdir(CURRENT_DIR)    
    return(partition, labels, wav_scp)


class FSDDDataFromDir(data.Dataset):
    def __init__(self, utt_ids, labels, wav_scp, transform=None):
        self.labels = labels
        self.utt_ids = utt_ids
        self.transform = transform
        self.wav_scp = wav_scp

    def __len__(self):
        return(len(self.utt_ids))

    def __getitem__(self, index):
        utt_id = self.utt_ids[index]        
        X, sr = librosa.load(self.wav_scp[utt_id])
        y = self.labels[utt_id]
        # we want to output a torch tensor
        X = torch.from_numpy(X)
        sample = {'X': X, 'y': y}

        if self.transform:
            sample = self.transform(sample)

        return sample


