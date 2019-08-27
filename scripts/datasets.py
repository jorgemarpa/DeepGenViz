import os, sys
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import preprocessing
from src.utils import normalize, return_dt

local_root = '/Users/jorgetil/Google Drive/Colab_Notebooks/data'
colab_root = '/content/drive/My Drive/Colab_Notebooks/data'
exalearn_root = '/home/jorgemarpa/data'

## load pkl synthetic light-curve files to numpy array
class EROS2_lightcurves(Dataset):
    def __init__(self, use_time=True, use_err=True,
                 norm=True, folded=True, machine='local', seq_len=150):
        """EROS light curves data loader"""
        if machine == 'local':
            root = local_root
        elif machine == 'colab':
            root = colab_root
        elif machine == 'exalearn':
            root = exalearn_root
        else:
            print('Wrong machine, please select loca, colab or exalearn')
            sys.exit()
        if not folded:
            data_path = ('%s/time_series/real' % (root) +
                        '/EROS2_lcs_B_meta_snr5_augmented_trim%i.pkl'
                        % (seq_len))
        else:
            data_path = ('%s/time_series/real' % (root) +
                        '/EROS2_lcs_B_meta_snr5_augmented_folded_trim%i.pkl'
                        % (seq_len))
        print('Loading from:\n', data_path)
        self.aux = joblib.load(data_path)
        self.lcs = self.aux['lcs'].astype(np.float32)
        self.meta = self.aux['meta']
        del self.aux
        self.labels = self.meta['Type'].values
        ## integer encoding of labels
        self.label_int_enc = preprocessing.LabelEncoder()
        self.label_int_enc.fit(self.labels)
        self.labels_int = self.label_int_enc.transform(self.labels)
        ## one-hot encoding of labels
        self.label_onehot_enc = preprocessing.OneHotEncoder(sparse=False,
                                                            categories='auto',
                                                            dtype=np.float32)
        self.label_onehot_enc.fit(self.labels.reshape(-1, 1))
        self.labels_onehot = self.label_onehot_enc.transform(self.labels.reshape(-1, 1))

        if use_time and not use_err:
            self.lcs = self.lcs[:, :, 0:2]
        if not use_time and not use_err:
            self.lcs = self.lcs[:, :, 1:2]

        if not 'folded' in data_path:
            self.lcs = return_dt(self.lcs)
        if norm:
            self.lcs = normalize(self.lcs, n_feat=self.lcs.shape[2],
                                scale_to=[.0001, .9999], norm_time=use_time)


    def __getitem__(self, index):
        lc = self.lcs[index]
        label = self.labels[index]
        meta = self.meta.iloc[index]
        onehot = self.labels_onehot[index]
        return lc, label, onehot


    def __len__(self):
        return len(self.lcs)


    def drop_class(self, name):
        idx = np.where(self.labels != name)[0]
        self.lcs = self.lcs[idx]
        self.labels = self.labels[idx]
        self.meta = self.meta.iloc[idx]
        self.labels_onehot = self.labels_onehot[idx]
        self.labels_int = self.labels_int[idx]


    def only_class(self, name):
        idx = np.where(self.labels == name)[0]
        self.lcs = self.lcs[idx]
        self.labels = self.labels[idx]
        self.meta = self.meta.iloc[idx]
        self.labels_onehot = self.labels_onehot[idx]
        self.labels_int = self.labels_int[idx]


    def class_value_counts(self):
        print(self.meta.Type.value_counts())


    def get_dataloader(self, batch_size=32, shuffle=True,
                       test_split=0.2, random_seed=42):

        if test_split == 0.:
            train_loader = DataLoader(self, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=True)
            test_loader = None
        else:
            # Creating data indices for training and test splits:
            dataset_size = len(self)
            indices = list(range(dataset_size))
            split = int(np.floor(test_split * dataset_size))
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            train_indices, test_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)

            train_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=train_sampler, drop_last=True)
            test_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=test_sampler, drop_last=True)

        return train_loader, test_loader


class Synt_lightcurves(Dataset):
    def __init__(self, use_time=True, use_err=True,
                 norm=True, colab=False, seq_len=100):
        """EROS light curves data loader"""
        if colab:
            root = colab_root
        else:
            root = local_root
        if not folded:
            data_path = ('%s/time_series/synthetic' % (root) +
                         '/sine_nsamples%i_seqlength%i_nbands%i_nsig%i_timespan%i_SNR%i_f0%s.npy'
                         % (28000, 100, 1, 1, 4, 3, 'narrow'))
        print('Loading from:\n', data_path)
        self.aux = np.load(data_path, allow_pickle=True).item()
        self.lcs = self.aux['samples'].astype(np.float32)
        self.meta = aux['periods']
        del self.aux
        self.labels = np.random.randint(0, 5, self.lcs.shape[0])
        self.labels_onehot = pd.get_dummies(self.meta['Type']).values
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.labels)
        self.labels_int = self.label_encoder.transform(self.labels)

        if use_time and not use_err:
            self.lcs = self.lcs[:, :, 0:2]
        if not use_time and not use_err:
            self.lcs = self.lcs[:, :, 1:2]

        self.lcs = return_dt(self.lcs)
        if norm:
            self.lcs = normalize(self.lcs, n_feat=self.lcs.shape[2],
                                scale_to=[-.9,.9], norm_time=use_time)


    def __getitem__(self, index):
        lc = self.lcs[index]
        label = self.labels[index]
        meta = self.meta.iloc[index]
        label_enc = self.labels_int[index]
        return lc, label_enc


    def __len__(self):
        return len(self.lcs)


    def class_value_counts(self):
        print(self.meta.Type.value_counts())


    def get_dataloader(self, batch_size=32, shuffle=True,
                       test_split=0.2, random_seed=42):

        if test_split == 0.:
            train_loader = DataLoader(self, batch_size=batch_size,
                                      shuffle=shuffle, drop_last=True)
            test_loader = None
        else:
            # Creating data indices for training and test splits:
            dataset_size = len(self)
            indices = list(range(dataset_size))
            split = int(np.floor(test_split * dataset_size))
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            train_indices, test_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)

            train_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=train_sampler, drop_last=True)
            test_loader = DataLoader(self, batch_size=batch_size,
                                      sampler=test_sampler, drop_last=True)

        return train_loader, test_loader
