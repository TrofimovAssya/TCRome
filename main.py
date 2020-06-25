#!/usr/bin/env python
import torch
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
    parser.add_argument('--dataset', choices=['tcr','hla_tcr',
                                              'binary_hla_tcr'], default='tcr', help='Which dataset to use.')
    parser.add_argument('--transform', default=True,help='log10(exp+1)')
    parser.add_argument('--nb-patient', default=5,type=int, help='nb of different patients')
    parser.add_argument('--tcr-size', default=27,type=int, help='length of the TCR sequence')
    parser.add_argument('--hla-size', default=34,type=int, help='length of the HLA sequence')
    parser.add_argument('--nb-kmer', default=1000,type=int, help='nb of different kmers')
    parser.add_argument('--cache', default=0,type=int, help='cache prefix for the dataset')
    parser.add_argument('--nb-tcr-to-sample', default=10000,type=int, help='nb of TCR to sample')
    # Model specific options
    parser.add_argument('--tcr-conv-layers-sizes', default=[20,1,18], type=int, nargs='+', help='TCR-Conv net config.')
    parser.add_argument('--hla-conv-layers-sizes', default=[20,1,25], type=int, nargs='+', help='HLA-Conv net config.')
    parser.add_argument('--mlp-layers-size', default=[250, 75, 50, 25, 10], type=int, nargs='+', help='MLP config')
    parser.add_argument('--emb_size', default=10, type=int, help='The size of the embeddings.')
    parser.add_argument('--loss', choices=['NLL', 'MSE'], default = 'MSE', help='The cost function to use')
    parser.add_argument('--weight-decay', default=1e-5, type=float, help='Weight decay parameter.')
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

    dataset = datasets.get_dataset(opt,exp_dir)

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

    if opt.model == 'allseq' or opt.model == 'allseq_bin':
        valid_list = np.load('/u/trofimov/Emerson/processed_data/valid_list.npy')
        loss_dict['valid_losses'] = []

    # The training.
    print ("Start training.")
    #monitoring and predictions
    for t in range(epoch, opt.epoch):
        loss_dict = monitoring.update_loss_dict(loss_dict,start = True)

        for no_b, mini in enumerate(dataset):
            loss_dict['train_losses_epoch'] = []
            if not opt.model == 'TCRonly':
                loss_dict['valid_losses_epoch'] = []


            if opt.model == 'TCRonly':

                y_pred, my_model, targets = training.TCRonly_batch(mini,opt,my_model)
                loss = criterion(y_pred, targets)
                loss_save = loss.data.cpu().numpy().reshape(1,)[0]
                loss_dict['train_losses_epoch'].append(loss_save)

                if no_b % 5 == 0:
                    print (f"Doing epoch{t},examples{no_b}/{len(dataset)}.Loss:{loss_save}")

                # Saving the emb
                np.save(os.path.join(exp_dir, 'pixel_epoch_{}'.format(t)),my_model.emb_1.weight.cpu().data.numpy())


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                kmerembs = my_model.get_embeddings(inputs_k, inputs_s)[0].squeeze()
                np.save(f'{exp_dir}/kmer_embs/kmer_embs_batch_{no_b}',kmerembs.cpu().data.numpy())

            elif opt.model == 'allseq':

                inputs_k,inputs_h1, inputs_h2, inputs_h3, inputs_h4, targets = training.allseq_batch(mini,opt)
                y_pred = my_model(inputs_k,inputs_h1, inputs_h2, inputs_h3,
                                  inputs_h4).float()

                loss = criterion(y_pred, targets)
                loss_save = loss.data.cpu().numpy().reshape(1,)[0]
                if no_b in valid_list:
                    loss_dict['valid_losses_epoch'].append(loss_save)
                    print (f"Validation error {t},examples{no_b}/{len(dataset)}.Loss:{loss_save}")

                elif no_b % 5 == 0:
                    loss_dict['train_losses_epoch'].append(loss_save)
                    print (f"Doing epoch {t},examples{no_b}/{len(dataset)}.Loss:{loss_save}")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                batch_number = dataset.dataset.data[no_b]
                kmerembs = my_model.get_embeddings(inputs_k, inputs_h1,
                                                   inputs_h2, inputs_h3,
                                                   inputs_h4)
                kmerembs1 = kmerembs[0].squeeze()
                bn = batch_number[0]
                np.save(f'{exp_dir}/tcr_embs/tcr_embs_batch_{bn}',kmerembs1.cpu().data.numpy())

                for i in range(4):
                    kmerembs1 = kmerembs[i+1].squeeze()
                    kmerembs1 = kmerembs1[0]
                    np.save(f'{exp_dir}/hla_embs/hla_embs_batch_{bn}_h{i+1}',kmerembs1.cpu().data.numpy())


                kmerembs1 = my_model.hla_representation
                kmerembs1 = kmerembs1[0].squeeze()
                np.save(f'{exp_dir}/hla_embs/ppl_embs_batch_{bn}',kmerembs1.cpu().data.numpy())




            elif opt.model == 'allseq_bin':

                inputs_k, inputs_h1, inputs_h2, inputs_h3, inputs_h4, targets = training.binallseq_batch(mini,opt)
                y_pred = my_model(inputs_k,inputs_h1, inputs_h2, inputs_h3,
                                  inputs_h4).float()

                loss = criterion(y_pred, targets)
                loss_save = loss.data.cpu().numpy().reshape(1,)[0]

                if no_b in valid_list:
                    loss_dict['valid_losses_epoch'].append(loss_save)
                    print (f"Validation error {t},examples{no_b}/{len(dataset)}.Loss:{loss_save}")

                elif no_b % 5 == 0:
                    loss_dict['train_losses_epoch'].append(loss_save)
                    print (f"Doing epoch {t},examples{no_b}/{len(dataset)}.Loss:{loss_save}")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                batch_number = dataset.dataset.data[no_b]
                kmerembs = my_model.get_embeddings(inputs_k, inputs_h1,
                                                   inputs_h2, inputs_h3,
                                                   inputs_h4)
                kmerembs1 = kmerembs[0].squeeze()
                bn = batch_number[0]
                true_size = int(kmerembs1.shape[0]/2)
                np.save(f'{exp_dir}/tcr_embs/tcr_embs_batch_{bn}',kmerembs1.cpu().data.numpy()[:true_size])

                for i in range(4):
                    kmerembs1 = kmerembs[i+1].squeeze()
                    kmerembs1 = kmerembs1[0]
                    np.save(f'{exp_dir}/hla_embs/hla_embs_batch_{bn}_h{i+1}',kmerembs1.cpu().data.numpy()[:true_size])


                kmerembs1 = my_model.hla_representation
                kmerembs1 = kmerembs1[0].squeeze()
                np.save(f'{exp_dir}/hla_embs/ppl_embs_batch_{bn}',kmerembs1.cpu().data.numpy()[:true_size])


        print ("Saving the model...")
        if opt.model=='allseq_bin' or opt.model=='allseq':
            validation_scores = loss_dict['valid_losses_epoch']
        else:
            validation_scores = None

        monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)
        monitoring.update_loss_dict(loss_dict, start=False)
        monitoring.save_loss(loss_dict,exp_dir)
        if t % opt.plot_frequency==0:
            monitoring.plot_training_curve(exp_dir, loss_dict)



    print ('Finished training! Starting evaluations')
    tcr_rep_dir = f'{exp_dir}/tcr_embs'
    patient_to_index = f'data/hla_for_model_eval/pt_names.csv'
    original_data_dir = f'/u/trofimov/Emerson/original'

    
    nb_patients = 15

    evaluations.evaluate_model(opt, my_model ,exp_dir, tcr_rep_dir, patient_to_index, 
                          original_data_dir, validation_scores, nb_patients,
                           train_on_index=0)


if __name__ == '__main__':
    main()
