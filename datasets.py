from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py
import pdb
from collections import OrderedDict
import shutil
import pandas as pd


class TCRDataset(Dataset):
    """TCR abundance dataset"""

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy', nb_patient = 5, nb_kmer = 1000):
        self.root_dir = root_dir
        data_path = os.path.join(root_dir, data_file)
        self.data = pd.read_csv(data_path, header=None)
        self.data = list(self.data[0])
        self.nb_patient = nb_patient
        self.nb_kmer = nb_kmer
        print (self.nb_kmer)
        print (self.nb_patient)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = self.data[idx]
        fnames = os.listdir('cached_dataset')
        if not f'{self.cache}_{idx}_tcr_gd.npy' in fnames:
            tcr = np.load(f'{self.root_dir}/{idx}_tcr_gd.npy')
            tcr = tcr[:self.nb_tcr_to_sample]
            np.save(f'cached_dataset/{self.cache}_{idx}_tcr_gd.npy',tcr)
        else:
            tcr = np.load(f'cached_dataset/{self.cache}_{idx}_tcr_gd.npy')


        sample = np.load(f'{self.root_dir}/{idx}_sample.npy')
        label = np.load(f'{self.root_dir}/{idx}_freq_log10.npy')
        sample = [sample, tcr, label]

        return sample

    def input_size(self):
        return self.nb_patient, self.nb_kmer

    def extra_info(self):
        info = OrderedDict()
        return info

class TCRHLADataset(Dataset):
    """ dataset"""

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy'):
        self.root_dir = root_dir
        data_path = os.path.join(root_dir, data_file)
        self.data = pd.read_csv(data_path, header=None)
        self.data = list(self.data[0])
        self.nb_patient = 10
        self.nb_kmer = 10
        print (self.nb_kmer)
        print (self.nb_patient)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = self.data[idx]
        fnames = os.listdir('cached_dataset')
        if not f'{self.cache}_{idx}_tcr_gd.npy' in fnames:
            tcr = np.load(f'{self.root_dir}/{idx}_tcr_gd.npy')
            tcr = tcr[:self.nb_tcr_to_sample]
            np.save(f'cached_dataset/{self.cache}_{idx}_tcr_gd.npy',tcr)
        else:
            tcr = np.load(f'cached_dataset/{self.cache}_{idx}_tcr_gd.npy')

        h1 = np.load(f'{self.root_dir}/{idx}_h1.npy')
        h2 = np.load(f'{self.root_dir}/{idx}_h2.npy')
        h3 = np.load(f'{self.root_dir}/{idx}_h3.npy')
        h4 = np.load(f'{self.root_dir}/{idx}_h4.npy')
        label = np.load(f'{self.root_dir}/{idx}_freq_log10.npy')
        label = (10**label-np.mean(10**label))/(np.std(10**label))
        label = (label-np.min(label))/(np.max(label)-np.min(label))
        sample = [tcr,h1,h2,h3,h4, label]

        return sample

    def transform_kmerseq_table(self, X_kmer):
        X_kmer = list(X_kmer)
        out_kmers = np.zeros((len(X_kmer), len(X_kmer[0])  , 4 ))

        for kmer in X_kmer:
            out_kmers[X_kmer.index(kmer)] = self.get_kmer_onehot(kmer)
        return np.array(out_kmers)

    def input_size(self):
        return self.nb_patient, self.nb_kmer

    def extra_info(self):
        info = OrderedDict()
        return info


class BinaryTCRDataset(Dataset):
    """Binary TCR presence dataset"""

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy',
                 nb_tcr_to_sample = 10000, cache='123abc'):
        self.root_dir = root_dir
        self.cache = cache
        data_path = os.path.join(root_dir, data_file)
        self.data = np.load(data_path)
        self.nb_patient = 10
        self.nb_kmer = 10
        self.nb_tcr_to_sample = int(nb_tcr_to_sample)
        print (self.nb_kmer)
        print (self.nb_patient)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = self.data[idx]
        idx, idx_n = idx[0], idx[1]
        fnames = os.listdir('cached_dataset')
        if not f'{self.cache}_{idx}_tcr_gd.npy' in fnames:
            tcr = np.load(f'{self.root_dir}/{idx}_tcr_gd.npy')
            tcr = tcr[:self.nb_tcr_to_sample]
            np.save(f'cached_dataset/{self.cache}_{idx}_tcr_gd.npy',tcr)
        else:
            tcr = np.load(f'cached_dataset/{self.cache}_{idx}_tcr_gd.npy')

        if not f'{self.cache}_{idx_n}_tcr_gd.npy' in fnames:
            tcr_n = np.load(f'{self.root_dir}/{idx_n}_tcr_gd.npy')
            tcr_n = tcr_n[:self.nb_tcr_to_sample]
            np.save(f'cached_dataset/{self.cache}_{idx_n}_tcr_gd.npy',tcr_n)
        else:
            tcr_n = np.load(f'cached_dataset/{self.cache}_{idx_n}_tcr_gd.npy')
        h1 = np.load(f'{self.root_dir}/{idx}_h1.npy')
        h2 = np.load(f'{self.root_dir}/{idx}_h2.npy')
        h3 = np.load(f'{self.root_dir}/{idx}_h3.npy')
        h4 = np.load(f'{self.root_dir}/{idx}_h4.npy')
        tcr_total = np.vstack((tcr,tcr_n))
        sample = [tcr_total,h1,h2,h3,h4]

        return sample

    def input_size(self):
        return self.nb_patient, self.nb_kmer

    def extra_info(self):
        info = OrderedDict()
        return info




def get_dataset(opt, exp_dir):

    if opt.dataset == 'tcr':
        dataset = TCRDataset(root_dir=opt.data_dir, save_dir =exp_dir,data_file = opt.data_file, nb_patient = opt.nb_patient, nb_kmer = opt.nb_kmer)
    elif opt.dataset == 'hla_tcr':
        dataset = TCRHLADataset(root_dir=opt.data_dir, save_dir =exp_dir,data_file = opt.data_file)
    elif opt.dataset == 'binary_hla_tcr':
        dataset = BinaryTCRDataset(root_dir=opt.data_dir,
                                   save_dir =exp_dir,data_file = opt.data_file,
                                   nb_tcr_to_sample = opt.nb_tcr_to_sample,
                                   cache = opt.cache)
    else:
        raise NotImplementedError()

    dataloader = DataLoader(dataset, batch_size=opt.batch_size,shuffle=False,num_workers=1)
    return dataloader

def preprocessing(data_dir,fname):
    pass

