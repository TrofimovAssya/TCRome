#!/usr/bin/env python
import torch
import json
import pdb
import numpy as np
from torch.autograd import Variable
import os
import argparse
import datasets
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
    parser.add_argument('--dataset', default='simple')
    parser.add_argument('--cache', default=0, help='cache prefix for the dataset')
    # Model specific options
    parser.add_argument('--weight-decay', default=0, type=float, help='Weight decay parameter.')
    parser.add_argument('--model', choices=['logreg_benchmark'])
    parser.add_argument('--cpu', action='store_true', help='True if no gpu to be used')
    parser.add_argument('--nb-tcr-to-sample', default=200, type=int, help='nbtcr')
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

    # creating the dataset
    print ("Getting the dataset...")
    if not 'cached_dataset' in os.listdir('.'):
        os.mkdir('cached_dataset')

    dataset = datasets.get_dataset(opt,exp_dir)

    # Creating a model
    print ("Getting the model...")
    my_model, optimizer, epoch, opt = monitoring.load_checkpoint(exp_dir, opt, dataset.dataset.input_size(), )

    # Training optimizer and stuff
    criterion = torch.nn.BCELoss()

    if not opt.cpu:
        print ("Putting the model on gpu...")
        my_model.cuda(opt.gpu_selection)



    loss_dict = {}
    loss_dict['train_losses'] = []

    valid_list = np.load('/u/trofimov/Emerson/processed_data/valid_list.npy')
    loss_dict['valid_losses'] = []




    # The training.
    print ("Start training.")
    #monitoring and predictions
    for t in range(epoch, opt.epoch):
        loss_dict = monitoring.update_loss_dict(loss_dict,start = True)

        for no_b, mini in enumerate(dataset):

            inputs_x,targets  = mini[0], mini[1]
            inputs_x = Variable(inputs_x, requires_grad=False).float()
            targets = Variable(targets, requires_grad=False).float()
            if not opt.cpu:
                inputs_x = inputs_x.cuda(opt.gpu_selection)
                targets = targets.cuda(opt.gpu_selection)

            y_pred = my_model(inputs_x)
            loss = criterion(y_pred, targets)
            loss_save = loss.data.cpu().numpy().reshape(1,)[0]
            if no_b in valid_list:
                loss_dict['valid_losses_epoch'].append(loss_save)
                print (f"Validation error {t},examples{no_b}/{len(dataset)}.Loss:{loss_save}")

            else:
                loss_dict['train_losses_epoch'].append(loss_save)
                print (f"Doing epoch {t},examples{no_b}/{len(dataset)}.Loss:{loss_save}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        validation_scores = loss_dict['valid_losses_epoch']

        monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)
        monitoring.update_loss_dict(loss_dict, start=False)
        monitoring.save_loss(loss_dict,exp_dir)
        if t % opt.plot_frequency==0:
            monitoring.plot_training_curve(exp_dir, loss_dict)

if __name__ == '__main__':
    main()
