#!/usr/bin/env python
import torch
import json
import pdb
import numpy as np
from torch.autograd import Variable
import os
import argparse
import datasets
import old_datasets
import models
import pickle
import time
import random
import monitoring
import training
import evaluations
#
def build_parser():
    parser = argparse.ArgumentParser(description="")

    ### Hyperparameter options
    parser.add_argument('--epoch', default=10, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--seed', default=260389, type=int, help='Seed for random initialization and stuff.')
    parser.add_argument('--batch-size', default=1, type=int, help="The batch size.")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

    ### Dataset specific options
    parser.add_argument('--data-dir', default='./data/', help='The folder contaning the dataset.')
    parser.add_argument('--data-file', default='.', help='The data file with the dataset.')
    parser.add_argument('--dataset', choices=['tcr','hla_tcr',
                                              'binary_hla_tcr'], default='tcr', help='Which dataset to use.')
    parser.add_argument('--tenth', default=0, type=int, help='test set only - fraction')
    parser.add_argument('--transform', default=True,help='log10(exp+1)')
    parser.add_argument('--nb-patient', default=5,type=int, help='nb of different patients')
    parser.add_argument('--tcr-size', default=27,type=int, help='length of the TCR sequence')
    parser.add_argument('--hla-size', default=34,type=int, help='length of the HLA sequence')
    parser.add_argument('--nb-kmer', default=1000,type=int, help='nb of different kmers')
    parser.add_argument('--cache', default=0, help='cache prefix for the dataset')
    parser.add_argument('--nb-tcr-to-sample', default=10000,type=int, help='nb of TCR to sample')
    # Model specific options
    parser.add_argument('--tcr-conv-layers-sizes', default=[20,1,18], type=int, nargs='+', help='TCR-Conv net config.')
    parser.add_argument('--hla-conv-layers-sizes', default=[20,1,25], type=int, nargs='+', help='HLA-Conv net config.')


    parser.add_argument('--mlp-layers-size', default=[250, 75, 50, 25, 10], type=int, nargs='+', help='MLP config')
    parser.add_argument('--emb_size', default=10, type=int, help='The size of the embeddings.')
    parser.add_argument('--loss', choices=['NLL', 'MSE'], default = 'MSE', help='The cost function to use')
    parser.add_argument('--weight-decay', default=0, type=float, help='Weight decay parameter.')
    parser.add_argument('--model', choices=['RNN','TCRonly',
                                            'allseq','allseq_bin'], default='TCRonly', help='Which model to use.')
    parser.add_argument('--cpu', action='store_true', help='True if no gpu to be used')
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")
    parser.add_argument('--gpu-selection', type=int, default=0, help="gpu selection")


    # Monitoring options
    parser.add_argument('--plot-frequency', default=1, type=int, help='frequency (in nb epochs at which to generate training curve')
    parser.add_argument('--load-folder', help='The folder where to load and restart the training.')
    parser.add_argument('--save-dir', default='./testing123/', help='The folder where everything will be saved.')

    return parser

def parse_args(argv):

    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv

    return opt

def main(argv=None):

    opt = parse_args(argv)
    # TODO: set the seed
    seed = opt.seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    if opt.cache==0:
        opt.cache = random.getrandbits(128)

    exp_dir = opt.load_folder
    if exp_dir is None: # we create a new folder if we don't load.
        exp_dir = monitoring.create_experiment_folder(opt)

    if opt.model == 'RNN':
        print ('This model is deprecated - please use TCRonly from now on')
    # creating the dataset
    print ("Getting the dataset...")
    if not 'cached_dataset' in os.listdir('.'):
        os.mkdir('cached_dataset')

    opt.dataset = 'binary_test'
    tenth=opt.tenth
    dataset = old_datasets.get_dataset(opt,exp_dir,test=True)
    #dataset = datasets.get_dataset(opt,exp_dir,tenth=opt.tenth)

    # Creating a model
    print ("Getting the model...")
    my_model, optimizer, epoch, opt = monitoring.load_checkpoint(exp_dir, opt, dataset.dataset.input_size(), )

    criterion = torch.nn.MSELoss()
    # Training optimizer and stuff
    if opt.loss == 'NLL' or opt.model=='allseq_bin':
        criterion = torch.nn.NLLLoss()
        criterion = torch.nn.BCELoss()



    if not 'tcr_embs' in os.listdir(exp_dir):
        if opt.model == 'TCRonly':
            os.mkdir(f'{exp_dir}/tcr_embs/')
        elif opt.model == 'allseq' or opt.model == 'allseq_bin':
                os.mkdir(f'{exp_dir}/tcr_embs/')
                os.mkdir(f'{exp_dir}/hla_embs/')


    if not opt.cpu:
        print ("Putting the model on gpu...")
        my_model.cuda(opt.gpu_selection)



    loss_dict = {}
    loss_dict['train_losses'] = []

    def estimate_batch_accuracy(y,yhat):
        return np.sum([i==j for i,j in zip(y,yhat)])/y.shape[0]

    if opt.model == 'allseq' or opt.model == 'allseq_bin':
        valid_list = np.load('/u/trofimov/Emerson/processed_data/valid_list.npy')
        loss_dict['valid_losses'] = []




    # The training.
    print ("Getting the likelihood")
    os.mkdir(f'{exp_dir}/tenth{tenth}_preds_100/')
    #monitoring and predictions
    for t in range(1):
        loss_dict = monitoring.update_loss_dict(loss_dict,start = True)
        if opt.model == 'allseq_bin':
            good = 0

        for no_b, mini in enumerate(dataset):


            if opt.model == 'TCRonly':

                y_pred, my_model, targets = training.TCRonly_batch(mini,opt,my_model)
                np.save(f'{exp_dir}/preds_100/likelihood_batch{no_b}.npy',y_pred.data.cpu().numpy())

                if no_b % 5 == 0:
                    print (f"Doing epoch{t},examples{no_b}/{len(dataset)}")

                # Saving the emb


            elif opt.model == 'allseq':

                inputs_k,inputs_h1, inputs_h2, inputs_h3, inputs_h4, targets = training.allseq_batch(mini,opt)
                y_pred = my_model(inputs_k,inputs_h1, inputs_h2, inputs_h3,
                                  inputs_h4).float()

                np.save(f'{exp_dir}/preds_100/likelihood_batch{no_b}.npy',y_pred.data.cpu().numpy())
                batch_number = dataset.dataset.data[no_b]
                bn = batch_number[0]
                np.save(f'{exp_dir}/preds_100/likelihood_batch{bn}.npy',y_pred.data.cpu().numpy())

                if no_b % 5 == 0:
                    print (f"Doing epoch {t},examples{no_b}/{len(dataset)}")



            elif opt.model == 'allseq_bin':

                inputs_k, inputs_h1, inputs_h2, inputs_h3, inputs_h4, targets = training.binallseq_batch(mini,opt)
                y_pred = my_model(inputs_k,inputs_h1, inputs_h2, inputs_h3,
                                  inputs_h4).float()
                batch_number = dataset.dataset.data[no_b]
                bn = batch_number[0]
                np.save(f'{exp_dir}/tenth{tenth}_preds_100/likelihood_batch{bn}.npy',y_pred.data.cpu().numpy())
                if no_b % 5 == 0:
                    print (f"Doing epoch {t},examples{no_b}/{len(dataset)}")



if __name__ == '__main__':
    main()
