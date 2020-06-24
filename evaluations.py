import os
import numpy as np
import random
import torch
import shutil
import models
import datetime
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_model(opt, model):

    ### calculating the performance for various tasks of  the trained model
    ### MHC cluster correlation
    pcc = evaluate_mhc_representations(mhc_reprez)
    pass

def evaluate_mhc_representations(mhc_reprez,
                                 mhclist='data/hla_for_model_eval/mhc_eval_list_names.csv/',
                                 evalist='data/hla_for_model_eval/small_set_hla_MHCcluster_dist',
                                 nb_pairs = 1000):

    mhcclust = pd.read_csv(evalist, sep='\t',index_col=0)
    mhclist = pd.read_csv(mhclist,header=None)[0]

    ### reordering mhcclust so that the indices orders match the list
    reorder_mhcclust = np.zeros(mhcclust.shape)
    temp = np.array(mhcclust)
    mhcclust_cols = list(mhcclust.columns)
    for hla1 in mhclist:
        h1 = mhcclust_cols.index(hla1)
        i = mhclist.index(hla1)
        for hla2 in mhclist:
            h2 = mhcclust_cols.index(hla2)
            j = mhclist.index(hla2)
            reorder_mhcclust[i,j] = temp[h1,h2]


    ### pickling a random number of pairs of hlas
    indices = np.random.choice(range(mhcclust.shape[0]), nb_pairs)

    mhcdist = []
    fedist = []

    for i,j in zip(indices[:-1],indices[1:]):
        mhcdist.append(reorder_mhcclust[i,j])
        fedist.append(t.iloc[i,j])
        ed = np.linalg.norm((mhc_reprez[i]-mhc_reprez[j]))

    pcc = np.corrcoef(mhcdist,fedist)[0.1]

    plt.plot(mhcdist, fedist)

    plt.xlabel('MHCclust distance')
    plt.ylabel('Embedding distance')

    img_path = os.path.join(exp_dir,f'mhc_eval_plot.png')
    plt.savefig(img_path)

    return pcc



def evaluate_jgene_bypatient(on_umap=True):
    pass

def evaluate_jgene_staticset(on_umap=True):
    pass

def get_umap():
    pass




def create_experiment_folder(opt):

    params = vars(opt).copy()
    params = str(params)

    # create a experiment folder
    this_hash = random.getrandbits(128)
    this_hash = "%032x" % this_hash # in hex

    exp_dir = os.path.join(opt.save_dir, this_hash)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    f = open(os.path.join(exp_dir,'run_parameters'), 'w')
    f.write(params+'\n')
    f.close()
    print (vars(opt))
    print (f"Saving the everything in {exp_dir}")

    with open(os.path.join(opt.save_dir, 'experiment_table.txt'), 'a') as f:
        f.write('time: {} folder: {} experiment: {}\n'.format(datetime.datetime.now(), this_hash, params))

    return exp_dir

