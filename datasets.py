from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pdb
import pickle
from collections import OrderedDict
import pandas as pd

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
        cached_files = os.listdir('cached_dataset')
        data_path = os.path.join(root_dir, data_file)
        self.nb_patient = int(nb_patient)
        self.data = np.load(data_path)
        cutoffix = np.where(self.data[:,0]==500)[0][0]
        self.data = self.data[:cutoffix]
        print (f'Keeping the first {cutoffix} patients ')
        self.old_data = np.load(data_path)
        self.old_data = self.old_data[:cutoffix]
        self.nb_tcr_to_sample = int(nb_tcr_to_sample)


        ### Deciding if we are using the one-hot or gd dataset
        if 'oh' in self.cache:
            self.prefix='oh'
        else:
            self.prefix='gd'

        ### Checking if the dataset is cached already
        if f'{self.cache}cached_list.npy' in cached_files:
            print ('Found a cached dataset matching the cache ID!')
            self.data = np.load(f'cached_dataset/{self.cache}cached_list.npy')
            self.fnames_dict = pickle.load(open(f'cached_dataset/{self.cache}_fnamedict.p', 'rb'))
            for ix in self.old_data[:,1]:
                if not ix in self.fnames_dict:
                    print (f'Doing additional patient {ix}')
                    tcr = self.load_file(ix)
                    this_fnames_list = self.save_into_batches(ix, tcr,
                                                              self.nb_tcr_to_sample)
                    self.fnames_dict[ix] = this_fnames_list
        else:
            print ('No cached dataset found...')
            print ('Processing dataset...')
            new_datalist = np.ones((1,2))
            fnames_dict = {}
            for x in range(self.data.shape[0]):
                ix_all = self.data[x]
                ix = ix_all[0]
                print (f'Loading patient {ix} data...')
                tcr = self.load_file(ix)
                this_fnames_list = self.save_into_batches(ix, tcr, self.nb_tcr_to_sample)
                ### adding the newly cached data  filenames to the dictionary
                fnames_dict[ix] = this_fnames_list
                #import pdb;pdb.set_trace()
                new_datalist = np.vstack((new_datalist,
                           np.tile(ix_all,len(this_fnames_list)).reshape(len(this_fnames_list),2)))
            new_datalist = new_datalist[1:]
            self.fnames_dict = fnames_dict

            for ix in self.old_data[:,1]:
                if not ix in self.fnames_dict:
                    print (f'Doing additional patient {ix}')
                    tcr = self.load_file(ix)
                    this_fnames_list = self.save_into_batches(ix, tcr,
                                                              self.nb_tcr_to_sample)
                    fnames_dict[ix] = this_fnames_list

            self.data = new_datalist
            pickle.dump(self.fnames_dict,
                        open(f'cached_dataset/{self.cache}_fnamedict.p', 'wb'))
            np.save(f'cached_dataset/{self.cache}cached_list.npy', self.data)
            print ('Finished loading and caching data!')
        import pdb; pdb.set_trace()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        ### The negative one should sample a random TCR batch from all the
        ## corresponding negatives
        ix = int(ix)
        idx = self.data[ix]
        ### getting the negative and positive examples
        idx, idx_n = idx[0], idx[1]
        ### determining the batch number
        start_ix = np.where(self.data[:,0]==idx)[0][0]
        bnumber = ix-start_ix
        ### determining the filenames for neg and po examples
        idx_fname = self.fnames_dict[idx][bnumber]
        idx_n = int(idx_n)
        idx = int(idx)
        if bnumber>(len(self.fnames_dict[idx_n])-1):
            idx_n_fname = np.random.choice(self.fnames_dict[idx_n])
        else:
            idx_n_fname = self.fnames_dict[idx_n][bnumber]

        tcr = np.load(f'cached_dataset/{idx_fname}')
        tcr_n = np.load(f'cached_dataset/{idx_n_fname}')

        sizes = [tcr.shape[0], tcr_n.shape[0]]
        idx = int(idx)
        h1 = np.load(f'{self.root_dir}/{idx}_h1.npy')
        h2 = np.load(f'{self.root_dir}/{idx}_h2.npy')
        h3 = np.load(f'{self.root_dir}/{idx}_h3.npy')
        h4 = np.load(f'{self.root_dir}/{idx}_h4.npy')
        ### stacking the negative and the positive examples. 
        ### The label will be created later 
        tcr_total = np.vstack((tcr,tcr_n))
        sample = [tcr_total,h1,h2,h3,h4, sizes]

        return sample

    def load_file(self,ix):
        tcr = np.load(f'{self.root_dir}/{ix}_tcr_{self.prefix}.npy')
        tcr = tcr[np.random.permutation(np.arange(tcr.shape[0]))]
        return tcr



    def save_into_batches(self,ptix, tcr, bsize):
        ### this function will split the whole TCRset for a patient into
        ### batches of the same size and save them to file.
        ### It will also return a list of cached filenames to load later.
        count = 0
        fname_list = []
        for i in range(0,tcr.shape[0], bsize):
            print (f'Processing batch {i}/{tcr.shape[0]}')
            batch = tcr[i:i+bsize]
            batch_fname = f'{self.cache}_{ptix}_tcr_{self.prefix}_b{count}.npy'
            np.save(f'cached_dataset/{batch_fname}',batch)
            count+=1
            fname_list.append(batch_fname)
        return fname_list


    def input_size(self):
        ### possibly deprecated
        ### TODO: check if this is still needed!
        return self.nb_patient

    def extra_info(self):
        ### possibly deprecated
        ### TODO: check if this is still needed!
        info = OrderedDict()
        return info

class BinaryTestTCRDataset(Dataset):

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
        cached_files = os.listdir('cached_dataset')
        data_path = os.path.join(root_dir, data_file)
        self.nb_patient = int(nb_patient)
        self.data = np.load(data_path)
        cutoffix = np.where(self.data[:,0]>500)[0][0]
        self.data = self.data[cutoffix:]
        print (f'Keeping the last {self.data.shape[0]} patients ')
        self.old_data = np.load(data_path)
        self.old_data = self.old_data[cutoffix:]
        self.nb_tcr_to_sample = int(nb_tcr_to_sample)


        ### Deciding if we are using the one-hot or gd dataset
        if 'oh' in self.cache:
            self.prefix='oh'
        else:
            self.prefix='gd'

        ### Checking if the dataset is cached already
        if f'{self.cache}valid_cached_list.npy' in cached_files:
            print ('Found a cached validation dataset matching the cache ID!')
            self.data = np.load(f'cached_dataset/{self.cache}valid_cached_list.npy')
            self.fnames_dict = pickle.load(open(f'cached_dataset/{self.cache}valid_fnamedict.p', 'rb'))
            for ix in self.old_data.reshape(self.old_data.shape[0]*self.old_data.shape[1],1)[:,0]:
                if not ix in self.fnames_dict:
                    print (f'Doing additional patient {ix}')
                    tcr = self.load_file(ix)
                    this_fnames_list = self.save_into_batches(ix, tcr,
                                                              self.nb_tcr_to_sample)
                    self.fnames_dict[ix] = this_fnames_list
        else:
            print ('No cached validation dataset found...')
            print ('Processing dataset...')
            new_datalist = np.ones((1,2))
            fnames_dict = {}
            for x in range(self.data.shape[0]):
                ix_all = self.data[x]
                ix = ix_all[0]
                print (f'Loading patient {ix} data...')
                tcr = self.load_file(ix)
                this_fnames_list = self.save_into_batches(ix, tcr, self.nb_tcr_to_sample)
                ### adding the newly cached data  filenames to the dictionary
                fnames_dict[ix] = this_fnames_list
                #import pdb;pdb.set_trace()
                new_datalist = np.vstack((new_datalist,
                           np.tile(ix_all,len(this_fnames_list)).reshape(len(this_fnames_list),2)))
            new_datalist = new_datalist[1:]
            self.fnames_dict = fnames_dict

            for ix in self.old_data[:,1]:
                if not ix in self.fnames_dict:
                    print (f'Doing additional patient {ix}')
                    tcr = self.load_file(ix)
                    this_fnames_list = self.save_into_batches(ix, tcr,
                                                              self.nb_tcr_to_sample)
                    fnames_dict[ix] = this_fnames_list

            self.data = new_datalist
        pickle.dump(self.fnames_dict,
                    open(f'cached_dataset/{self.cache}valid_fnamedict.p', 'wb'))
        np.save(f'cached_dataset/{self.cache}valid_cached_list.npy', self.data)
        print ('Finished loading and caching validation data!')
        import pdb; pdb.set_trace()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        ### The negative one should sample a random TCR batch from all the
        ## corresponding negatives
        ix = int(ix)
        idx = self.data[ix]
        ### getting the negative and positive examples
        idx, idx_n = idx[0], idx[1]
        ### determining the batch number
        start_ix = np.where(self.data[:,0]==idx)[0][0]
        bnumber = ix-start_ix
        ### determining the filenames for neg and po examples
        idx_fname = self.fnames_dict[idx][bnumber]
        idx_n = int(idx_n)
        idx = int(idx)
        if bnumber>(len(self.fnames_dict[idx_n])-1):
            idx_n_fname = np.random.choice(self.fnames_dict[idx_n])
        else:
            idx_n_fname = self.fnames_dict[idx_n][bnumber]

        tcr = np.load(f'cached_dataset/{idx_fname}')
        tcr_n = np.load(f'cached_dataset/{idx_n_fname}')

        sizes = [tcr.shape[0], tcr_n.shape[0]]
        idx = int(idx)
        h1 = np.load(f'{self.root_dir}/{idx}_h1.npy')
        h2 = np.load(f'{self.root_dir}/{idx}_h2.npy')
        h3 = np.load(f'{self.root_dir}/{idx}_h3.npy')
        h4 = np.load(f'{self.root_dir}/{idx}_h4.npy')
        ### stacking the negative and the positive examples. 
        ### The label will be created later 
        tcr_total = np.vstack((tcr,tcr_n))
        sample = [tcr_total,h1,h2,h3,h4, sizes]

        return sample

    def load_file(self,ix):
        tcr = np.load(f'{self.root_dir}/{ix}_tcr_{self.prefix}.npy')
        tcr = tcr[np.random.permutation(np.arange(tcr.shape[0]))]
        return tcr



    def save_into_batches(self,ptix, tcr, bsize):
        ### this function will split the whole TCRset for a patient into
        ### batches of the same size and save them to file.
        ### It will also return a list of cached filenames to load later.
        count = 0
        fname_list = []
        for i in range(0,tcr.shape[0], bsize):
            print (f'Processing batch {i}/{tcr.shape[0]}')
            batch = tcr[i:i+bsize]
            batch_fname = f'{self.cache}_{ptix}_tcr_{self.prefix}_b{count}.npy'
            np.save(f'cached_dataset/{batch_fname}',batch)
            count+=1
            fname_list.append(batch_fname)
        return fname_list


    def input_size(self):
        ### possibly deprecated
        ### TODO: check if this is still needed!
        return self.nb_patient

    def extra_info(self):
        ### possibly deprecated
        ### TODO: check if this is still needed!
        info = OrderedDict()
        return info


class BinaryTCRDatasetLargeRandom(Dataset):

    """Binary TCR presence dataset
    Modification: train a certain number of samples with max TCR
    Train the rest with a random number of TCRs
    Contains negative/positive examples.
    Negative examples are real TCR taken from individuals
    that do not share HLA alleles with the current individual.

    The dataset caches files to avoid re-loading and reconstructing
    """

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy',
                 nb_tcr_to_sample = 10000, nb_patient = 10, cache='123abc'):
        self.root_dir = root_dir
        self.cache = str(cache)
        self.cache = f'{self.cache}_randomlarge'
        data_path = os.path.join(root_dir, data_file)
        self.nb_patient = int(nb_patient)
        self.data = np.load(data_path)[:self.nb_patient]
        self.nb_kmer = 10
        self.nb_tcr_to_sample = int(nb_tcr_to_sample)
        print (self.nb_kmer)
        print (self.nb_patient)
        if 'oh' in self.cache:
            self.prefix='oh'
        else:
            self.prefix='gd'
        #if test:
        #    self.cache = f'bottom{self.cache}'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx%10==0:
            maxkeep=True
        else:
            maxkeep=False

        idx = self.data[idx]
        idx, idx_n = idx[0], idx[1]
        fnames = os.listdir('cached_dataset')
        if not f'{self.cache}_{idx}_tcr_{self.prefix}.npy' in fnames:
            tcr = np.load(f'{self.root_dir}/{idx}_tcr_{self.prefix}.npy')
            ### only keeping a certain number of TCR (the most abundant)
            ### TODO: this could be changed eventually
            if 'bottom' in self.cache:
                tcr = tcr[-self.nb_tcr_to_sample:]
            elif maxkeep:
                tcr = tcr[:100000]
            else:
                keeprand = np.random.permutation(np.arange(tcr.shape[0]))[:self.nb_tcr_to_sample]
                tcr = tcr[keeprand]
            #tcr/=np.max(tcr)
            np.save(f'cached_dataset/{self.cache}_{idx}_tcr_{self.prefix}.npy',tcr)
        else:
            tcr = np.load(f'cached_dataset/{self.cache}_{idx}_tcr_{self.prefix}.npy')

        if not f'{self.cache}_{idx_n}_tcr_{self.prefix}.npy' in fnames:
            tcr_n = np.load(f'{self.root_dir}/{idx_n}_tcr_{self.prefix}.npy')
            if 'bottom' in self.cache:
                tcr_n = tcr_n[-self.nb_tcr_to_sample:]
            elif maxkeep:
                tcr_n = tcr_n[:100000]
            else:
                keeprand = np.random.permutation(np.arange(tcr_n.shape[0]))[:self.nb_tcr_to_sample]
                tcr_n = tcr_n[keeprand]
            #tcr_n/=np.max(tcr_n)
            np.save(f'cached_dataset/{self.cache}_{idx_n}_tcr_{self.prefix}.npy',tcr_n)
        else:
            tcr_n = np.load(f'cached_dataset/{self.cache}_{idx_n}_tcr_{self.prefix}.npy')
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


class TestBinaryTCRDataset(Dataset):

    """Test Binary TCR presence dataset
    Contains negative/positive examples.
    Negative examples are real TCR taken from individuals
    that do not share HLA alleles with the current individual.

    The dataset caches files to avoid re-loading and reconstructing
    """

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy',
                 nb_tcr_to_sample = 10000, nb_patient = 10, cache='123abc',
                 tenth = 0, group='test'):
        self.root_dir = root_dir
        self.cache = str(cache)
        self.group = group
        data_path = os.path.join(root_dir, data_file)
        self.nb_patient = int(nb_patient)
        if self.group == 'test':
            self.data = np.load(data_path)[self.nb_patient:self.nb_patient+4]
        elif self.group == 'same':
            self.data = np.load(data_path)#[:100]
        elif self.group == 'thome':
            self.data = np.load(data_path)
        self.nb_kmer = 10
        self.nb_tcr_to_sample = int(nb_tcr_to_sample)
        self.tenth = tenth
        print (self.nb_kmer)
        print (self.nb_patient)
        if 'oh' in self.cache:
            self.prefix='oh'
        else:
            self.prefix='gd'


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = self.data[idx]
        idx, idx_n = idx[0], idx[1]

        print (f'Loading patient {idx}')
        fnames = os.listdir('cached_dataset')

        if not f'{self.cache}_{self.group}_{self.tenth}_{idx}_tcr_{self.prefix}.npy' in fnames:
            print ('File not cached! Loading...')
            tcr = np.load(f'{self.root_dir}/{idx}_tcr_{self.prefix}.npy')
            step = int(tcr.shape[0]/10)
            for i in range(10):
                print('Doing partition {i}')
                start = step*i
                tcr = tcr[start:start+self.nb_tcr_to_sample]
                np.save(f'{self.cache}_{self.group}_{i}_{idx}_tcr_{self.prefix}.npy', tcr)
        tcr = np.load(f'{self.cache}_{self.group}_{self.tenth}_{idx}_tcr_{self.prefix}.npy')



        if not f'{self.cache}_{self.group}_{self.tenth}_{idx_n}_tcr_{self.prefix}.npy' in fnames:
            print ('File not cached! Loading...')
            tcr_n = np.load(f'{self.root_dir}/{idx_n}_tcr_{self.prefix}.npy')
            step = int(tcr_n.shape[0]/10)
            for i in range(10):
                print('Doing partition {i}')
                start = step*i
                tcr_n = tcr_n[start:start+self.nb_tcr_to_sample]
                np.save(f'{self.cache}_{self.group}_{i}_{idx_n}_tcr_{self.prefix}.npy', tcr_n)
        tcr_n = np.load(f'{self.cache}_{self.group}_{self.tenth}_{idx_n}_tcr_{self.prefix}.npy')


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


class BinaryTCRDatasetSmallFull(Dataset):

    """Binary TCR presence dataset
    Modification: train a small  number of samples with max TCR
    Contains negative/positive examples.
    Negative examples are real TCR taken from individuals
    that do not share HLA alleles with the current individual.

    The dataset caches files to avoid re-loading and reconstructing
    """

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy',
                 nb_tcr_to_sample = 10000, nb_patient = 10, cache='123abc'):
        self.root_dir = root_dir
        self.cache = str(cache)
        self.cache = f'{self.cache}_randomlarge'
        data_path = os.path.join(root_dir, data_file)
        self.nb_patient = int(nb_patient)
        self.data = np.load(data_path)[:self.nb_patient]
        self.nb_kmer = 10
        self.nb_tcr_to_sample = int(nb_tcr_to_sample)
        print (self.nb_kmer)
        print (self.nb_patient)
        if 'oh' in self.cache:
            self.prefix='oh'
        else:
            self.prefix='gd'
        #if test:
        #    self.cache = f'bottom{self.cache}'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        idx = self.data[idx]
        idx, idx_n = idx[0], idx[1]
        fnames = os.listdir('cached_dataset')
        if not f'{self.cache}_{idx}_tcr_{self.prefix}.npy' in fnames:
            tcr = np.load(f'{self.root_dir}/{idx}_tcr_{self.prefix}.npy')
            ### only keeping a certain number of TCR (the most abundant)
            ### TODO: this could be changed eventually
            if 'bottom' in self.cache:
                tcr = tcr[-self.nb_tcr_to_sample:]
            else:
                tcr = tcr[:125000]
            #tcr/=np.max(tcr)
            np.save(f'cached_dataset/{self.cache}_{idx}_tcr_{self.prefix}.npy',tcr)
        else:
            tcr = np.load(f'cached_dataset/{self.cache}_{idx}_tcr_{self.prefix}.npy')

        if not f'{self.cache}_{idx_n}_tcr_{self.prefix}.npy' in fnames:
            tcr_n = np.load(f'{self.root_dir}/{idx_n}_tcr_{self.prefix}.npy')
            if 'bottom' in self.cache:
                tcr_n = tcr_n[-self.nb_tcr_to_sample:]
            else:
                tcr_n = tcr_n[:125000]
            #tcr_n/=np.max(tcr_n)
            np.save(f'cached_dataset/{self.cache}_{idx_n}_tcr_{self.prefix}.npy',tcr_n)
        else:
            tcr_n = np.load(f'cached_dataset/{self.cache}_{idx_n}_tcr_{self.prefix}.npy')
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

class SimpleModelDatset(Dataset):
    """
    This dataset is for the logistic regression benchmarking model
    """

    def __init__(self,root_dir='.',save_dir='.', data_file='data.npy',
                 nb_tcr_to_sample = 10000, nb_patient = 10, cache='123abc'):
        self.root_dir = root_dir
        self.cache = str(cache)
        self.cache = f'{self.cache}_lr_simple'
        data_path = os.path.join(root_dir, data_file)
        self.nb_patient = int(nb_patient)
        self.data = np.load(data_path)[:self.nb_patient]
        self.nb_kmer = 10
        self.nb_tcr_to_sample = int(nb_tcr_to_sample)
        print (self.nb_kmer)
        print (self.nb_patient)
        if 'oh' in self.cache:
            self.prefix='oh'
        else:
            self.prefix='gd'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        idx = self.data[idx]
        idx, idx_n = idx[0], idx[1]
        fnames = os.listdir('cached_dataset')
        if not f'{self.cache}_{idx}_tcr_{self.prefix}.npy' in fnames:
            tcr = np.load(f'{self.root_dir}/{idx}_tcr_{self.prefix}.npy')
            ### only keeping a certain number of TCR (the most abundant)
            ### TODO: this could be changed eventually
            if 'bottom' in self.cache:
                tcr = tcr[-self.nb_tcr_to_sample:]
            else:
                tcr = tcr[:self.nb_tcr_to_sample]

            tcr_n = np.load(f'{self.root_dir}/{idx_n}_tcr_{self.prefix}.npy')

            if 'bottom' in self.cache:
                tcr_n = tcr_n[-self.nb_tcr_to_sample:]
            else:
                tcr_n = tcr_n[:self.nb_tcr_to_sample]

            h1 = np.load(f'{self.root_dir}/{idx}_onehot_hla.npy')
            tcr_total = np.vstack((tcr,tcr_n))
            tcr_total = tcr_total.reshape(tcr_total.shape[0], 27*20)
            ### TODO: make this better using broadcasting?
            h_new = np.zeros((tcr_total.shape[0], h1.shape[0]))
            for i in range(h_new.shape[0]):
                h_new[i,:] = h1
            total_inputs = np.hstack((tcr_total,h_new))


            np.save(f'cached_dataset/{self.cache}_{idx}_tcr_{self.prefix}.npy',total_inputs)
        else:
            total_inputs = np.load(f'cached_dataset/{self.cache}_{idx}_tcr_{self.prefix}.npy')

        size = int(total_inputs.shape[0]/2)
        targets = np.zeros((total_inputs.shape[0], 2))
        targets[:size,1]+=1
        targets[size:,0]+=1
        return total_inputs, targets

    def input_size(self):
        ### possibly deprecated
        ### TODO: check if this is still needed!
        return self.nb_patient, self.nb_kmer

    def extra_info(self):
        ### possibly deprecated
        ### TODO: check if this is still needed!
        info = OrderedDict()
        return info



def get_dataset(opt, exp_dir, test=False, tenth=0):

    """
    Four datasets are implemented
    TCRDataset - the vanilla dataset [sample index, sequence] [abundance]
    TCRHLADataset - [hla1, hla2, hla3, hla4, tcrseq] [abundance]
    BinaryTCRDataset - [hla1, hla2, hla3, hla4, tcrseq] [present/absent]
    SimpleModelDataset - the dataset that is used for the simple model
    comparison

    """


    if opt.dataset == 'tcr':
        dataset = TCRDataset(root_dir=opt.data_dir, save_dir =exp_dir,data_file = opt.data_file, nb_patient = opt.nb_patient, nb_kmer = opt.nb_kmer)

    if opt.dataset == 'simple':
        dataset = SimpleModelDatset(root_dir=opt.data_dir,
                save_dir=exp_dir, data_file=opt.data_file,
                 nb_tcr_to_sample = opt.nb_tcr_to_sample, 
                 nb_patient = 500, cache=opt.cache)

    elif opt.dataset == 'hla_tcr':
        dataset = TCRHLADataset(root_dir=opt.data_dir, save_dir =exp_dir,data_file = opt.data_file)
    elif opt.dataset == 'binary_hla_tcr':
        dataset = BinaryTCRDataset(root_dir=opt.data_dir,
                                   save_dir =exp_dir,data_file = opt.data_file,
                                   nb_tcr_to_sample = opt.nb_tcr_to_sample,
                                   nb_patient = opt.nb_patient,
                                   cache = opt.cache)
    elif opt.dataset == 'binary_test':
        dataset = BinaryTestTCRDataset(root_dir=opt.data_dir,
                                   save_dir =exp_dir,data_file = opt.data_file,
                                   nb_tcr_to_sample = opt.nb_tcr_to_sample,
                                   nb_patient = opt.nb_patient,
                                   cache = opt.cache)
    elif opt.dataset == 'binary_same':
        dataset = TestBinaryTCRDataset(root_dir=opt.data_dir,
                                   save_dir =exp_dir,data_file = opt.data_file,
                                   nb_tcr_to_sample = opt.nb_tcr_to_sample,
                                   nb_patient = opt.nb_patient,
                                   cache = opt.cache, tenth=tenth, group='same')
    elif opt.dataset == 'binary_thome':
        dataset = TestBinaryTCRDataset(root_dir=opt.data_dir,
                                   save_dir =exp_dir,data_file = opt.data_file,
                                   nb_tcr_to_sample = opt.nb_tcr_to_sample,
                                   nb_patient = opt.nb_patient,
                                   cache = opt.cache, tenth=tenth, group='thome')
    elif opt.dataset == 'binary_rand':
        dataset = BinaryTCRDatasetLargeRandom(root_dir=opt.data_dir,
                                   save_dir =exp_dir,data_file = opt.data_file,
                                   nb_tcr_to_sample = opt.nb_tcr_to_sample,
                                   nb_patient = opt.nb_patient,
                                   cache = opt.cache)

    elif opt.dataset == 'binary_small':
        dataset = BinaryTCRDatasetSmallFull(root_dir=opt.data_dir,
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


