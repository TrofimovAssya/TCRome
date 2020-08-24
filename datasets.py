from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pdb
from collections import OrderedDict
import pandas as pd


class TCRDataset(Dataset):
    """TCR abundance dataset
    The dataset as defined in TLT paper
    added the sampling of TCRs for better comparison with the other models.
    """


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
            tcr/=np.max(tcr)
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
    """ TCR-HLA dataset
    Variation on the vanilla TCR dataset.
    the patient ix is replaced with the patient's HLA. 
    """

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy'):
        self.root_dir = root_dir
        data_path = os.path.join(root_dir, data_file)
        self.data = pd.read_csv(data_path, header=None)
        self.data = list(self.data[0])
        ### TODO: I think this is deprecated
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
            tcr/=np.max(tcr)
            np.save(f'cached_dataset/{self.cache}_{idx}_tcr_gd.npy',tcr)
        else:
            tcr = np.load(f'cached_dataset/{self.cache}_{idx}_tcr_gd.npy')

        h1 = np.load(f'{self.root_dir}/{idx}_h1.npy')
        h1/=np.max(h1)
        h2 = np.load(f'{self.root_dir}/{idx}_h2.npy')
        h2/=np.max(h2)
        h3 = np.load(f'{self.root_dir}/{idx}_h3.npy')
        h3/=np.max(h3)
        h4 = np.load(f'{self.root_dir}/{idx}_h4.npy')
        h4/=np.max(h4)
        label = np.load(f'{self.root_dir}/{idx}_freq_log10.npy')
        ### to Z-scores? for better prediction?
        ### TODO: add this as a parameter for the model
        label = (10**label-np.mean(10**label))/(np.std(10**label))
        label = (label-np.min(label))/(np.max(label)-np.min(label))
        sample = [tcr,h1,h2,h3,h4, label]

        return sample

    ### TODO: the next 2 function are possibly deprecated
    def input_size(self):
        return self.nb_patient, self.nb_kmer

    def extra_info(self):
        info = OrderedDict()
        return info


class BinaryTCRDataset(Dataset):
    """Binary TCR presence dataset
    Contains negative/positive examples.
    Negative examples are real TCR taken from individuals
    that do not share HLA alleles with the current individual.

    The dataset caches files to avoid re-loading and reconstructing
    """

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy',
                 nb_tcr_to_sample = 10000, nb_patient = 10, cache='123abc'):
        self.root_dir = root_dir
        self.cache = str(cache)
        data_path = os.path.join(root_dir, data_file)
        self.nb_patient = int(nb_patient)
        self.data = np.load(data_path)[:self.nb_patient]
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
            ### only keeping a certain number of TCR (the most abundant)
            ### TODO: this could be changed eventually
            if 'bottom' in self.cache:
                tcr = tcr[-self.nb_tcr_to_sample:]
            else:
                tcr = tcr[:self.nb_tcr_to_sample]
            #tcr/=np.max(tcr)
            np.save(f'cached_dataset/{self.cache}_{idx}_tcr_gd.npy',tcr)
        else:
            tcr = np.load(f'cached_dataset/{self.cache}_{idx}_tcr_gd.npy')

        if not f'{self.cache}_{idx_n}_tcr_gd.npy' in fnames:
            tcr_n = np.load(f'{self.root_dir}/{idx_n}_tcr_gd.npy')
            if 'bottom' in self.cache:
                tcr_n = tcr_n[-self.nb_tcr_to_sample:]
            else:
                tcr_n = tcr_n[:self.nb_tcr_to_sample]
            #tcr_n/=np.max(tcr_n)
            np.save(f'cached_dataset/{self.cache}_{idx_n}_tcr_gd.npy',tcr_n)
        else:
            tcr_n = np.load(f'cached_dataset/{self.cache}_{idx_n}_tcr_gd.npy')
        h1 = np.load(f'{self.root_dir}/{idx}_h1.npy')
        #h1/=np.max(h1)
        h2 = np.load(f'{self.root_dir}/{idx}_h2.npy')
        #h2/=np.max(h2)
        h3 = np.load(f'{self.root_dir}/{idx}_h3.npy')
        #h3/=np.max(h3)
        h4 = np.load(f'{self.root_dir}/{idx}_h4.npy')
        #h4/=np.max(h4)
        ### stacking the negative and the positive examples. 
        ### The label will be created later 
        tcr_total = np.vstack((tcr,tcr_n))
        sample = [tcr_total,h1,h2,h3,h4]

        return sample

    def input_size(self):
        ### possibly deprecated
        ### TODO: check if this is still needed!
        return self.nb_patient, self.nb_kmer

    def extra_info(self):
        ### possibly deprecated
        ### TODO: check if this is still needed!
        info = OrderedDict()
        return info


def get_dataset(opt, exp_dir):

    """
    Three datasets are implemented
    TCRDataset - the vanilla dataset [sample index, sequence] [abundance]
    TCRHLADataset - [hla1, hla2, hla3, hla4, tcrseq] [abundance]
    BinaryTCRDataset - [hla1, hla2, hla3, hla4, tcrseq] [present/absent]

    """


    if opt.dataset == 'tcr':
        dataset = TCRDataset(root_dir=opt.data_dir, save_dir =exp_dir,data_file = opt.data_file, nb_patient = opt.nb_patient, nb_kmer = opt.nb_kmer)
    elif opt.dataset == 'hla_tcr':
        dataset = TCRHLADataset(root_dir=opt.data_dir, save_dir =exp_dir,data_file = opt.data_file)
    elif opt.dataset == 'binary_hla_tcr':
        dataset = BinaryTCRDataset(root_dir=opt.data_dir,
                                   save_dir =exp_dir,data_file = opt.data_file,
                                   nb_tcr_to_sample = opt.nb_tcr_to_sample,
                                   nb_patient = opt.nb_patient,
                                   cache = opt.cache)
    else:
        raise NotImplementedError()

    dataloader = DataLoader(dataset, batch_size=opt.batch_size,shuffle=False,num_workers=1)
    return dataloader

def preprocessing(data_dir,fname):
    pass


