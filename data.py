# IMPORTS

import os
import random
from argsparser import get_args

from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
from tqdm import tqdm
import shutil
import wfdb
from wfdb import rdrecord, rdann, processing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# PARSER ARGUMENTS
parser = get_args()
# _data
URL = parser['url']
WRITE_PATH = parser['mitbih-preprocessed-path']
ROOT_PATH = parser['mitbih-path']
# _model
BATCH_SIZE = parser['batch-size']
EPOCHS = parser['epochs']
MODEL_NAME_SAVE = parser['model-name-save']
MODEL_NAME_LOAD = parser['model-name-load']
_L = parser['L']
_K = parser['K']
_g1 = parser['g1']
_cp = parser['cp']
_ap = parser['ap']
_g2 = parser['g2']
_n = parser['n']
_cSA = parser['cSA']
_aSA = parser['aSA']
_g3 = parser['g3']
_cb = parser['cb']
_ab = parser['ab']
_cSB = parser['cSB']
_aSB = parser['aSB']
_gB = parser['gB']
_n_classes = parser['n_classes']


# DATA

PATIENT_IDS = ['100','101','102','103','104','105','106','107',
           '108','109','111','112','113','114','115','116',
           '117','118','119','121','122','123','124','200',
           '201','202','203','205','207','208','209','210',
           '212','213','214','215','217','219','220','221',
           '222','223','228','230','231','232','233','234']

MAP = {'N':['N','L','R','B'],
       'S':['A','a','j','S','e','j','n'],
       'V':['V','r','E'],
       'F':['F'],
       'Q':['Q','?','f','/'],
      }

LABEL_TO_INT = dict(zip(MAP, range(len(MAP))))

AAMI = ['N','L','R','B','A','a','j','S','V','r','F','e','j','n','E','f','/','Q','?']


# Arguments from .ini file
def get_args():
    
    config = configparser.ConfigParser()
    config.read('config.ini')

    parser = {}    
    parser['mitbih-path'] = config.get('DATA', 'MIT-BIH-PATH')
    parser['mitbih-preprocessed-path'] = config.get('DATA', 'MIT-BIH-PREPROCESSED-PATH')
    parser['url'] = config.get('DATA', 'URL')
    
    return parser


# Downloading Raw Data
def download_and_unzip(url, extract_to='.'):
    
    print('Downloading and unzipping MIT-BIH dataset ...')

    if not os.path.exists(ROOT_PATH):
        http_response = urlopen(url)
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path=extract_to)
        
    else:
        print("MIT-BIH folder already exists at: mit-bih-arrhythmia-database-1.0.0")
        

# Preprocessing
def preprocessor(read_from, write_to, patients=PATIENT_IDS, fs=360, insize=360, overwrite=False, wipe_original=False):

    if not os.path.exists(WRITE_PATH):
        os.makedirs(WRITE_PATH)
         
    for num in patients:

        if os.path.exists(f'{write_to}/{num}_labels.csv') \
        and os.path.exists(f'{write_to}/{num}_beats.csv') \
        and not overwrite:
            # print(f'PATIENT NUMBER {num} [already processed]', end=' ')
            continue

        else:
            print(f'\nPATIENT NUMBER {num}')

            y = []
            beat_l2 = []

            record = wfdb.rdrecord(f'{read_from}/{num}')
            l2 = preprocessing.scale(np.nan_to_num(record.p_signal[:,0]))

            qrs = processing.XQRS(sig = l2,fs = fs)
            qrs.detect()
            peaks = qrs.qrs_inds


            for peak in tqdm(peaks[1:-1]):
                start,end = peak-insize//2, peak+insize//2
                ann = wfdb.rdann(f'{read_from}/{num}',extension='atr', sampfrom = start, sampto = end, 
                                 return_label_elements=['symbol'])

                annsymbol = ann.symbol
                if len(annsymbol) == 1 and annsymbol[0] in AAMI:
                    for cl, an in MAP.items():
                        if annsymbol[0] in an:
                            y.append(cl)
                            beat_l2.append(l2[start:end])


            # check if all signals are of size {insize}
            idx, beats = zip(*[(i,beat) for i, beat in enumerate(beat_l2) if beat.size == insize])
            beats, labels = pd.DataFrame(beats), pd.DataFrame([y[i] for i in idx])

            beats_path = os.path.join(write_to, f'{num}_beats.csv')
            labels_path = os.path.join(write_to, f'{num}_labels.csv')

            # write to write_folder_path
            beats.to_csv(beats_path, index=False)
            labels.to_csv(labels_path, index=False)   
                
                
    # we no longer need the original MIT-BIH folder
    if wipe_original and os.path.exists(read_from):
        
        shutil.rmtree(read_from)


# Read the .csv files that preprocessor() created
def reader(num, read_folder_path):
    
    beats, labels = pd.read_csv(f'{read_folder_path}/{num}_beats.csv'), pd.read_csv(f'{read_folder_path}/{num}_labels.csv')
    signals = [beats.loc[i].to_numpy() for i in range(len(beats))]
    labels = [LABEL_TO_INT[l] for l in labels['0'].to_numpy()]
    assert len(signals) == len(labels), "something's wrong, i can feel it"
    
    pairs = [(torch.tensor(sig).unsqueeze(0), l) for sig, l in zip(signals, labels)]
    
    return pairs


# Form training/testing pairs
def make_pairs(read_from, ids=PATIENT_IDS):
    
    pairs = []
    print("Forming Labeled Data Pairs ...")
    
    for num in tqdm(ids):
        pairs.extend( reader(num, read_from) )

    return pairs


# Torch Dataset
class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        sample, label = self.data[idx], self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label
    
    
# GATHERING & CLEANING DATA

def get_data():
    
    # _download raw data
    download_and_unzip(url=URL)

    # _preprocess data and write .csv files
    preprocessor(read_from=ROOT_PATH, write_to=WRITE_PATH)

    # _make training/testing pairs
    pairs = make_pairs(read_from=WRITE_PATH)

    # _train and test split [8:2], both splits maintain the same distribution of classes
    signals, labels = zip(*pairs)

    train_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,  
        stratify=labels,
        random_state=42
    )

    train_pairs, test_pairs = [pairs[i] for i in train_indices], [pairs[i] for i in test_indices]
    train_inputs, train_labels = zip(*train_pairs)
    test_inputs, test_labels = zip(*test_pairs)

    train_dataset = MyDataset(train_inputs, train_labels)
    test_dataset = MyDataset(test_inputs, test_labels)

    TRAINLOADER = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    TESTLOADER = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return train_dataset, test_dataset, TRAINLOADER, TESTLOADER