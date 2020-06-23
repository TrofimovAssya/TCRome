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

    pass

def evaluate_mhc_representations(mhc_reprez, evalist):
    pass

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

