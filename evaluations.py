import matplotlib
matplotlib.use('Agg')
import os
import json
import numpy as np
from torch.autograd import Variable
import random
import torch
import json
import models
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import umap
import time
from sklearn.neighbors import KNeighborsClassifier


def evaluate_model(opt, model, exp_dir, tcr_rep_dir, patient_to_index,
                   original_data_dir, validation_scores = None, nb_patients=15, train_on_index = 0,
                   on_umap=True):

    ### calculating the performance for various tasks of  the trained model
    ### MHC cluster correlation
    to_json = {}
    pcc = evaluate_mhc_representations(exp_dir, model, opt)
    to_json['mhc_pcc'] = pcc


    acc, results = evaluate_jgene_bypatient(tcr_rep_dir,
                                                patient_to_index,original_data_dir,
                                                nb_patients = 15,
                                                on_umap=on_umap,
                                                train_on_index = 0)

    to_json['tcr_knn_scores'] = {}

    if on_umap:
        to_json['tcr_knn_emb'] = acc[0]
        to_json['tcr_knn_umap'] = acc[1]
        to_json['tcr_knn_scores']['umap_scores'] = results[1]['accuracy']
        to_json['tcr_knn_scores']['emb_scores'] = results[0]['accuracy']
        to_json['tcr_knn_scores']['patient_names'] = results[0]['patient_names']
    else:
        to_json['tcr_knn_emb'] = acc
        to_json['tcr_knn_socres']['emb_scores'] = results['accuracy']
        to_json['tcr_knn_socres']['patient_names'] = results['patient_names']

    if not validation_scores==None:
        to_json['valid'] = np.mean('validation')

    return to_json



def evaluate_mhc_representations(exp_dir,
                                 my_model,
                                 opt,
                                 mhclist='data/hla_for_model_eval/mhc_eval_list_names.csv',
                                 evalist='data/hla_for_model_eval/small_set_hla_MHCcluster_dist',
                                 nb_pairs = 1000):

    mhcclust = pd.read_csv(evalist, sep='\t',index_col=0)
    mhclist = pd.read_csv(mhclist)['0']
    mhclist = list(mhclist)

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
    ###TODO: mhcreprez is only a directory. It should be the mhc
    mhcseq = np.load('data/hla_for_model_eval/mhc_eval_list_sequences.npy')
    mhcseq = torch.FloatTensor(mhcseq)
    mhcseq = Variable(mhcseq,requires_grad=False).float()
    if not opt.cpu:
        mhcseq = mhcseq.cuda(opt.gpu_selection)
    mhcseq = mhcseq.permute(0, 2, 1)


    mhc_reprez = my_model.encode_hla(mhcseq)
    mhc_reprez = mhc_reprez.cpu().data.numpy()
    ###representations instead

    for i,j in zip(indices[:-1],indices[1:]):
        mhcdist.append(reorder_mhcclust[i,j])
        ed = np.linalg.norm((mhc_reprez[i]-mhc_reprez[j]))
        fedist.append(ed)

    pcc = np.corrcoef(mhcdist,fedist)[0,1]

    plt.plot(mhcdist, fedist)

    plt.xlabel('MHCclust distance')
    plt.ylabel('Embedding distance')

    img_path = os.path.join(exp_dir,f'mhc_eval_plot.png')
    plt.savefig(img_path)

    return pcc



def evaluate_jgene_bypatient(tcr_rep_dir,
                             patient_to_index,
                             original_data_dir,
                             nb_patients = 15,
                             on_umap=True,
                            train_on_index = 0):

    tcr_rep_files = os.listdir(tcr_rep_dir)[:nb_patients]
    patient_to_index = pd.read_csv(patient_to_index, header=None)
    print (f'optimizing knn for the chosen patient:{tcr_rep_files[train_on_index]}')
    tcr_embs = np.load(f'{tcr_rep_dir}/{tcr_rep_files[train_on_index]}')
    pt_file_index = tcr_rep_files[train_on_index].split('_')
    pt_file_index = pt_file_index[-1].split('.')[0]
    pt_file_index = int(pt_file_index)
    data_file = list(patient_to_index[patient_to_index[0]==pt_file_index][1])
    if len(data_file)==0:
        import pdb;pdb.set_trace()
    else:
        data_file = data_file[0]
        data_file = f'{data_file}.tsv'
    labels = pd.read_csv(f'{original_data_dir}/{data_file}',sep='\t')
    labels = labels['j_gene']
    ### TODO: here we need to load the TCR seq set and decode to the original
    ### AA. From there, we need to get the j_genes!

    if on_umap:
        print ('getting umap')
        tcr_embs1 = get_umap(tcr_embs)
        clf1 = get_knn(tcr_embs1, labels, optimize_k=True, return_model=True)

    clf = get_knn(tcr_embs, labels, optimize_k=True, return_model=True)

    print ('looping through patients')
    scores = []
    patient_names = []
    scores_umap = []
    for patient in tcr_rep_files:
        print (f'Doing patient {tcr_rep_files.index(patient)}/{len(tcr_rep_files)}')
        pt_file_index = patient.split('_')
        pt_file_index = pt_file_index[-1].split('.')[0]
        pt_file_index = int(pt_file_index)
        data_file = list(patient_to_index[patient_to_index[0]==pt_file_index][1])
        if len(data_file)==0:
            import pdb;pdb.set_trace()
        else:
            data_file = data_file[0]
        data_file = f'{data_file}.tsv'
        labels = pd.read_csv(f'{original_data_dir}/{data_file}',sep='\t')
        labels = labels['j_gene']

        tcr_embs = np.load(f'{tcr_rep_dir}/{patient}')
        if not labels.shape[0]==tcr_embs.shape[0]:
            import pdb;pdb.set_trace()

        if on_umap:
            print ('getting umap')
            tcr_embs1 = get_umap(tcr_embs)
            scores_umap.append(clf1.score(tcr_embs1, labels))

        scores.append(clf.score(tcr_embs, labels))
        patient_names.append(data_file)

    perf = np.mean(scores)
    result = pd.DataFrame([scores, patient_names]).T
    result.columns = ['accuracy', 'patient_names']

    if on_umap:
        perf1 = np.mean(scores_umap)
        result1 = pd.DataFrame([scores_umap, patient_names])
        result1.columns = ['accuracy', 'patient_names']
        return [perf,perf1] [resul, result1]
    else:
        return perf, result



def evaluate_jgene_staticset(seqset,
                             on_umap=True):
    pass


def get_umap(emb):
    reducer = umap.UMAP(verbose=1)
    emb = reducer.fit_transform(emb)
    return emb



def get_knn(data, labels, chosenk = 5,
        optimize_k = False, nb_shuffles=20,return_model=False):


    if optimize_k:
        nneigh = []
        valid_perf = []
        shuffles = []
        for i in range(nb_shuffles):
            print (f'Shuffle #{i}')
            start = time.time()
            shuffles.append(i)
            index_shuffles = np.random.permutation(np.arange(data.shape[0]))
            split_80 = int(data.shape[0]*0.8)
            split_90 = int(data.shape[0]*0.9)
            train_ix, valid_ix, test_ix = index_shuffles[:split_80],index_shuffles[split_80:split_90], index_shuffles[split_90:]
            for k in [2,3,4,5,6,10,15,20,25,30,35,40,45,100,200]:
                clf = KNeighborsClassifier(n_neighbors=k)
                clf.fit(data[train_ix,:],labels[train_ix])
                perf = np.sum([i==j for i,j in
                               zip(clf.predict(data[valid_ix,:]),labels[valid_ix])])/valid_ix.shape[0]
                nneigh.append(k)
                valid_perf.append(perf)
                print (k)
            stop = time.time()
            elapsed = stop-start
            print (f'took {elapsed} seconds')
        result = pd.DataFrame([nneigh,valid_perf,shuffles]).T
        result.columns = ['n_neigh','perf','shuffle']
        #chosenk = list(result.iloc[np.argmax(result['perf']),:]['n_neigh'])[0]
        chosenk = result.iloc[np.argmax(result['perf']),:]['n_neigh']
        chosenk = int(chosenk)
    else:
        print ('splitting data')
        index_shuffles = np.random.permutation(np.arange(data.shape[0]))
        split_80 = int(data.shape[0]*0.8)
        split_90 = int(data.shape[0]*0.9)
        train_ix, valid_ix, test_ix = index_shuffles[:split_80],index_shuffles[split_80:split_90], index_shuffles[split_90:]

    clf = KNeighborsClassifier(n_neighbors=chosenk)
    clf.fit(data[np.hstack((train_ix,valid_ix)),:],labels[np.hstack((train_ix,valid_ix))])
    perf = np.sum([i==j for i,j in
                   zip(clf.predict(data[test_ix,:]),labels[test_ix])])/test_ix.shape[0]
    print (f'Final performance {perf}')
    if return_model:
        return clf 
    else:
        return perf





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

