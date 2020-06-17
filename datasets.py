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
        #indices_p = np.arange(self.data.shape[0])
        #indices_k = np.arange(self.data.shape[1])
        #self.X_data = np.transpose([np.tile(indices_k, len(indices_p)), np.repeat(indices_p, len(indices_k))])



        #self.root_dir = root_dir
        #self.X_data, self.Y_data = self.dataset_make(np.array(self.data),log_transform=False)
        #self.X_sample = self.X_data[:,1]
        #self.X_kmer = self.data.columns[self.X_data[:,0]]
        #self.X_kmer = self.transform_kmerseq_table(self.X_kmer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #fname_sample = f'{self.root_dir}/{self.data[idx]}_samples.npy'
        #fname_kmer = f'{self.root_dir}/{self.data[idx]}_clones.npy'
        #fname_label = f'{self.root_dir}/{self.data[idx]}_targets.npy'
        #sample = np.load(fname_sample)
        #kmer = np.load(fname_kmer)
        #kmer = kmer[np.tile(np.arange(kmer.shape[0]), self.nb_patient)]
        #label = np.load(fname_label)
        #label = np.log10(label)
        #label = np.log10(label+1)
        #kmer = np.load(f'{self.root_dir}/{idx}_tcr.npy')
        #kmer = np.load(f'{self.root_dir}/{idx}_tcr_oh.npy')
        #kmer = np.load(f'{self.root_dir}/{idx}_tcr_gd_sm.npy')
        idx = self.data[idx]
        kmer = np.load(f'{self.root_dir}/{idx}_tcr_gd.npy')
        sample = np.load(f'{self.root_dir}/{idx}_sample.npy')
        label = np.load(f'{self.root_dir}/{idx}_freq_log10.npy')
        sample = [sample, kmer, label]

        return sample

    def dataset_make(self, gene_exp, log_transform=False):
        indices_p1 = np.arange(gene_exp.shape[0])
        indices_g = np.arange(gene_exp.shape[1])
        X_data = np.transpose([np.tile(indices_g, len(indices_p1)), np.repeat(indices_p1, len(indices_g))])
        Y_data = gene_exp[X_data[:, 1], X_data[:, 0]]

        print (f"Total number of examples: {Y_data.shape} ")

        if log_transform:
            Y_data = np.log10(Y_data + 1)
        return X_data, Y_data

    def get_kmer_onehot(self,kmer):
        convert = {'A':3, 'G':1, 'C':2, 'T':0}
        result = np.zeros((len(kmer), 4))
        for i in range(len(kmer)):
            result[i,convert[kmer[i]]] +=1
        return result

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

class TCRHLADataset(Dataset):
    """Kmer abundance dataset"""

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy'):
        self.root_dir = root_dir
        data_path = os.path.join(root_dir, data_file)
        self.data = pd.read_csv(data_path, header=None)
        self.data = list(self.data[0])
        self.nb_patient = 10
        self.nb_kmer = 10
        print (self.nb_kmer)
        print (self.nb_patient)
        #indices_p = np.arange(self.data.shape[0])
        #indices_k = np.arange(self.data.shape[1])
        #self.X_data = np.transpose([np.tile(indices_k, len(indices_p)), np.repeat(indices_p, len(indices_k))])



        #self.root_dir = root_dir
        #self.X_data, self.Y_data = self.dataset_make(np.array(self.data),log_transform=False)
        #self.X_sample = self.X_data[:,1]
        #self.X_kmer = self.data.columns[self.X_data[:,0]]
        #self.X_kmer = self.transform_kmerseq_table(self.X_kmer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #fname_sample = f'{self.root_dir}/{self.data[idx]}_samples.npy'
        #fname_kmer = f'{self.root_dir}/{self.data[idx]}_clones.npy'
        #fname_label = f'{self.root_dir}/{self.data[idx]}_targets.npy'
        #sample = np.load(fname_sample)
        #kmer = np.load(fname_kmer)
        #kmer = kmer[np.tile(np.arange(kmer.shape[0]), self.nb_patient)]
        #label = np.load(fname_label)
        #label = np.log10(label)
        #label = np.log10(label+1)
        #kmer = np.load(f'{self.root_dir}/{idx}_tcr.npy')
        #kmer = np.load(f'{self.root_dir}/{idx}_tcr_oh.npy')
        #kmer = np.load(f'{self.root_dir}/{idx}_tcr_gd_sm.npy')
        idx = self.data[idx]
        kmer = np.load(f'{self.root_dir}/{idx}_tcr_gd.npy')
        h1 = np.load(f'{self.root_dir}/{idx}_h1.npy')
        h2 = np.load(f'{self.root_dir}/{idx}_h2.npy')
        h3 = np.load(f'{self.root_dir}/{idx}_h3.npy')
        h4 = np.load(f'{self.root_dir}/{idx}_h4.npy')
        #h1 = np.repeat(h1,kmer.shape[0])
        #h2 = np.repeat(h2,kmer.shape[0])
        #h3 = np.repeat(h3,kmer.shape[0])
        #h4 = np.repeat(h4,kmer.shape[0])
        label = np.load(f'{self.root_dir}/{idx}_freq_log10.npy')
        label = (10**label-np.mean(10**label))/(np.std(10**label))
        label = (label-np.min(label))/(np.max(label)-np.min(label))
        sample = [kmer,h1,h2,h3,h4, label]

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
        #self.data = self.data[:5,:]
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
        fnames = os.listdir('.')
        if not f'cached_dataset/{self.cache}_{idx}_tcr_gd.npy' in fnames:
            tcr = np.load(f'{self.root_dir}/{idx}_tcr_gd.npy')
            tcr = tcr[:self.nb_tcr_to_sample]
            np.save(f'cached_dataset/{self.cache}_{idx}_tcr_gd.npy',tcr)
        else:
            tcr = np.load(f'cached_dataset/{self.cache}_{idx}_tcr_gd.npy')

        if not f'cached_dataset/{self.cache}_{idx_n}_tcr_gd.npy' in fnames:
            tcr_n = np.load(f'{self.root_dir}/{idx_n}_tcr_gd.npy')
            tcr_n = tcr_n[:self.nb_tcr_to_sample]
            np.save(f'cached_dataset/{self.cache}_{idx_n}_tcr_gd.npy',tcr_n)
        else:
            tcr_n = np.load(f'cached_dataset/{self.cache}_{idx_n}_tcr_gd.npy')
        #tcr_n = np.load(f'{self.root_dir}/{idx_n}_tcr_gd.npy')
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

    #TODO: check the num_worker, might be important later on, for when we will use a bunch of big files.
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,shuffle=False,num_workers=1)
    return dataloader

def preprocessing(data_dir,fname):
    pass

