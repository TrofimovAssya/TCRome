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
    parser.add_argument('--cache', default=0,type=int, help='nb of different kmers')
    parser.add_argument('--nb-tcr-to-sample', default=10000,type=int, help='nb of TCR to sample')
    # Model specific options
    parser.add_argument('--tcr-conv-layers-sizes', default=[20,1,18], type=int, nargs='+', help='TCR-Conv net config.')
    parser.add_argument('--hla-conv-layers-sizes', default=[20,1,25], type=int, nargs='+', help='HLA-Conv net config.')
    parser.add_argument('--mlp-layers-size', default=[250, 75, 50, 25, 10], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--emb_size', default=2, type=int, help='The size of the embeddings.')
    parser.add_argument('--loss', choices=['NLL', 'MSE'], default = 'MSE', help='The cost function to use')

    parser.add_argument('--weight-decay', default=1e-5, type=float, help='The size of the embeddings.')
    parser.add_argument('--model', choices=['RNN','TCRonly',
                                            'allseq','allseq_bin'], default='TCRonly', help='Which model to use.')
    parser.add_argument('--cpu', action='store_true', help='If we want to run on cpu.') # TODO: should probably be cpu instead.
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")
    parser.add_argument('--gpu-selection', type=int, default=0, help="selectgpu")


    # Monitoring options
    parser.add_argument('--save-error', action='store_true', help='If we want to save the error for each tissue and each gene at every epoch.')
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

    if opt.model == 'TCRonly':
        os.mkdir(f'{exp_dir}/kmer_embs/')
    elif opt.model == 'allseq':
        os.mkdir(f'{exp_dir}/tcr_embs/')
        os.mkdir(f'{exp_dir}/pep_embs/')
    elif opt.model == 'allseq_bin':
        os.mkdir(f'{exp_dir}/tcr_embs/')
        os.mkdir(f'{exp_dir}/pep_embs/')


    if not opt.cpu:
        print ("Putting the model on gpu...")
        my_model.cuda(opt.gpu_selection)

    if opt.model == 'allseq' or opt.model == 'allseq_bin':
        valid_list = np.load('/u/trofimov/Emerson/processed_data/valid_list.npy')

    # The training.
    print ("Start training.")
    #monitoring and predictions
    for t in range(epoch, opt.epoch):

        start_timer = time.time()
        i=0
        for no_b, mini in enumerate(dataset):
            i+=1


            if opt.model == 'TCRonly':
                #inputs_s, inputs_k, targets = mini[0], mini[1], mini[2]
                #inputs_s = Variable(inputs_s, requires_grad=False).float()
                #inputs_k = Variable(inputs_k, requires_grad=False).float()
                #targets = Variable(targets, requires_grad=False).float()
                #if not opt.cpu:
                #    inputs_s = inputs_s.cuda(opt.gpu_selection)
                #    inputs_k = inputs_k.cuda(opt.gpu_selection)
                #    targets = targets.cuda(opt.gpu_selection)
                #inputs_k = inputs_k.squeeze().permute(0, 2, 1)
                #y_pred = my_model(inputs_k,inputs_s).float()
                #y_pred = y_pred.permute(1,0)
                y_pred, my_model, targets = training.TCRonly_batch(mini,opt,my_model)
                loss = criterion(y_pred, targets)
                if no_b % 5 == 0:
                    print (f"Doing epoch {t},examples{no_b}/{len(dataset)}.Loss:{loss.data.cpu().numpy().reshape(1,)[0]}")

                # Saving the emb
                np.save(os.path.join(exp_dir, 'pixel_epoch_{}'.format(t)),my_model.emb_1.weight.cpu().data.numpy())


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            elif opt.model == 'allseq':

                #inputs_k, inputs_h1, inputs_h2, inputs_h3, inputs_h4, targets = mini[0], mini[1], mini[2], mini[3], mini[4], mini[5]
                #inputs_h1 = inputs_h1.repeat(inputs_k.shape[1],1,1)
                #inputs_h2 = inputs_h2.repeat(inputs_k.shape[1],1,1)
                #inputs_h3 = inputs_h3.repeat(inputs_k.shape[1],1,1)
                #inputs_h4 = inputs_h4.repeat(inputs_k.shape[1],1,1)
                #inputs_k = Variable(inputs_k, requires_grad=False).float()
                #targets = Variable(targets, requires_grad=False).float()
                #inputs_h1 = Variable(inputs_h1, requires_grad=False).float()
                #inputs_h2 = Variable(inputs_h2, requires_grad=False).float()
                #inputs_h3 = Variable(inputs_h3, requires_grad=False).float()
                #inputs_h4 = Variable(inputs_h3, requires_grad=False).float()

                #if not opt.cpu:
                #    inputs_k = inputs_k.cuda(opt.gpu_selection)
                #    inputs_h1 = inputs_h1.cuda(opt.gpu_selection)
                #    inputs_h2 = inputs_h2.cuda(opt.gpu_selection)
                #    inputs_h3 = inputs_h3.cuda(opt.gpu_selection)
                #    inputs_h4 = inputs_h4.cuda(opt.gpu_selection)
                #    targets = targets.cuda(opt.gpu_selection)
                #inputs_k = inputs_k.squeeze().permute(0, 2, 1)
                #inputs_h1 = inputs_h1.squeeze().permute(0, 2, 1)
                #inputs_h2 = inputs_h2.squeeze().permute(0, 2, 1)
                #inputs_h3 = inputs_h3.squeeze().permute(0, 2, 1)
                #inputs_h4 = inputs_h4.squeeze().permute(0, 2, 1)
                inputs_k,inputs_h1, inputs_h2, inputs_h3, inputs_h4, targets = training.allseq_batch(mini,opt)
                y_pred = my_model(inputs_k,inputs_h1, inputs_h2, inputs_h3,
                                  inputs_h4).float()
                #y_pred = y_pred.permute(1,0)
                if no_b == 10:
                    print ('******')
                    print ((y_pred.data.cpu().numpy()))
                    print ((targets.data.cpu().numpy()))
                    print (np.std(y_pred.data.cpu().numpy()))
                    print ('******')

                #targets = torch.reshape(targets,(targets.shape[0],1))

                loss = criterion(y_pred, targets)
                if no_b in valid_list:
                    print (f"Validation error {t},examples{no_b}/{len(dataset)}.Loss:{loss.data.cpu().numpy().reshape(1,)[0]}")
                elif no_b % 5 == 0:
                    print (f"Doing epoch {t},examples{no_b}/{len(dataset)}.Loss:{loss.data.cpu().numpy().reshape(1,)[0]}")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            elif opt.model == 'allseq_bin':

                #inputs_k, inputs_h1, inputs_h2, inputs_h3, inputs_h4 = mini[0], mini[1], mini[2], mini[3], mini[4]

                #if inputs_h1.shape[1]>inputs_k.shape[1]:
                #    inputs_h1 = inputs_h1[:,:inputs_k.shape[1],:,:]
                #    inputs_h2 = inputs_h2[:,:inputs_k.shape[1],:,:]
                #    inputs_h3 = inputs_h3[:,:inputs_k.shape[1],:,:]
                #    inputs_h4 = inputs_h4[:,:inputs_k.shape[1],:,:]
                #elif inputs_h1.shape[1]>1:
                #    inputs_h1 = inputs_h1.repeat(inputs_k.shape[1],1,1)
                #    inputs_h2 = inputs_h2.repeat(inputs_k.shape[1],1,1)
                #    inputs_h3 = inputs_h3.repeat(inputs_k.shape[1],1,1)
                #    inputs_h4 = inputs_h4.repeat(inputs_k.shape[1],1,1)
                #inputs_k = Variable(inputs_k, requires_grad=False).float()
                #inputs_h1 = Variable(inputs_h1, requires_grad=False).float()
                #inputs_h2 = Variable(inputs_h2, requires_grad=False).float()
                #inputs_h3 = Variable(inputs_h3, requires_grad=False).float()
                #inputs_h4 = Variable(inputs_h3, requires_grad=False).float()
                #targets = np.zeros((inputs_k.shape[1],2))
                #size = int(inputs_k.shape[1]/2)
                #targets[:size,1]+=1
                #targets[size:,0]+=1
                #targets = torch.FloatTensor(targets)
                #targets = Variable(targets,requires_grad=False).float()

                #if not opt.cpu:
                #    inputs_k = inputs_k.cuda(opt.gpu_selection)
                #    inputs_h1 = inputs_h1.cuda(opt.gpu_selection)
                #    inputs_h2 = inputs_h2.cuda(opt.gpu_selection)
                #    inputs_h3 = inputs_h3.cuda(opt.gpu_selection)
                #    inputs_h4 = inputs_h4.cuda(opt.gpu_selection)
                #    targets = targets.cuda(opt.gpu_selection)
                #inputs_k = inputs_k.squeeze().permute(0, 2, 1)
                #inputs_h1 = inputs_h1.squeeze().permute(0, 2, 1)
                #inputs_h2 = inputs_h2.squeeze().permute(0, 2, 1)
                #inputs_h3 = inputs_h3.squeeze().permute(0, 2, 1)
                #inputs_h4 = inputs_h4.squeeze().permute(0, 2, 1)
                inputs_k, inputs_h1, inputs_h2, inputs_h3, inputs_h4, targets = training.binallseq_batch(mini,opt)
                y_pred = my_model(inputs_k,inputs_h1, inputs_h2, inputs_h3,
                                  inputs_h4).float()
                #import pdb;pdb.set_trace()
                #y_pred = y_pred.permute(1,0)
                #targets = torch.reshape(targets,(targets.shape[0],1))

                loss = criterion(y_pred, targets)
                if no_b in valid_list:
                    print (f"Validation error {t},examples{no_b}/{len(dataset)}.Loss:{loss.data.cpu().numpy().reshape(1,)[0]}")
                elif no_b % 5 == 0:
                    print (f"Doing epoch {t},examples{no_b}/{len(dataset)}.Loss:{loss.data.cpu().numpy().reshape(1,)[0]}")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


            # Zero gradients, perform a backward pass, and update the weights.
            if opt.model=='TCRonly':
                kmerembs = my_model.get_embeddings(inputs_k, inputs_s)[0].squeeze()
                np.save(f'{exp_dir}/kmer_embs/kmer_embs_batch_{no_b}',kmerembs.cpu().data.numpy())
            elif opt.model == 'allseq':
                batch_number = dataset.dataset.data[no_b]
                kmerembs = my_model.get_embeddings(inputs_k, inputs_h1,
                                                   inputs_h2, inputs_h3,
                                                   inputs_h4)
                kmerembs = kmerembs[0].squeeze()
                np.save(f'{exp_dir}/tcr_embs/tcr_embs_batch_{batch_number}',kmerembs.cpu().data.numpy())

                kmermembs = my_model.hla_representation
                kmerembs = kmerembs[0].squeeze()
                np.save(f'{exp_dir}/pep_embs/pep_embs_batch_{batch_number}',kmerembs.cpu().data.numpy())
            elif opt.model == 'allseq_bin':
                batch_number = dataset.dataset.data[no_b]
                kmerembs = my_model.get_embeddings(inputs_k, inputs_h1,
                                                   inputs_h2, inputs_h3,
                                                   inputs_h4)
                kmerembs = kmerembs[0].squeeze()
                bn = batch_number[0]
                np.save(f'{exp_dir}/tcr_embs/tcr_embs_batch_{bn}',kmerembs.cpu().data.numpy())

                kmermembs = my_model.hla_representation
                kmerembs = kmerembs[0].squeeze()
                np.save(f'{exp_dir}/pep_embs/pep_embs_batch_{bn}',kmerembs.cpu().data.numpy())


        print ("Saving the model...")
        monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)



if __name__ == '__main__':
    main()
