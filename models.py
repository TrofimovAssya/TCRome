import torch
import torch.nn.functional as F
from torch import nn
import argparse
from itertools import chain


class FactorizedCNN(nn.Module):

    def __init__(self, conv_layers_sizes = [20,10,15,10,5,12], 
        mlp_layers_size = [25,10], 
        nb_samples=1, emb_size=10, 
        tcr_input_size = 27,
        data_dir ='.'):
        super(FactorizedCNN, self).__init__()

        self.emb_size = emb_size
        self.sample = nb_samples
        self.tcr_input_size = tcr_input_size

        layers = []
        outsize = self.tcr_input_size

        for i in range(0,len(conv_layers_sizes),3):
            layer = nn.Conv1d(in_channels = conv_layers_sizes[i+0],out_channels = conv_layers_sizes[i+1],kernel_size = conv_layers_sizes[i+2],stride = 1)
            outsize = outsize-conv_layers_sizes[i+2]+1
            layers.append(layer)
            dim = [(conv_layers_sizes[i+1]*(outsize))]

        self.tcr_conv_stack = nn.ModuleList(layers)

        self.emb_1 = nn.Embedding(self.sample, emb_size)

        layers = []
        self.dim = [dim+emb_size] + mlp_layers_size 

        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(dim[-1], 1)


    def get_embeddings(self, x1, x2):

        kmer, patient = x1, x2
        for layer in self.tcr_conv_stack:
            kmer = layer(kmer)
        kemr = kmer.squeeze()
        patient = self.emb_1(patient.long())
        return kmer, patient

    def forward(self, x1, x2):

        emb_1, emb_2 = self.get_embeddings(x1, x2)
        emb_1 = emb_1.permute(1,0,2)
        emb_1 = emb_1.squeeze()
        emb_2 = emb_2.squeeze()

        if not emb_1.shape == emb_2.shape:
            import pdb; pdb.set_trace()
        mlp_input = torch.cat([emb_1, emb_2], 1)

        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)
        mlp_output = self.last_layer(mlp_input)

        return mlp_output



class AllSeqCNN(nn.Module):

    def __init__(self, tcr_conv_layers_sizes = [20,1,18], 
        hla_conv_layers_sizes = [20,1,25], 
        mlp_layers_size = [25,10],
        tcr_input_size = 27,
        hla_input_size = 34, 
        nb_samples=1, emb_size=10, data_dir ='.'):
        super(AllSeqCNN, self).__init__()

        self.emb_size = emb_size
        self.sample = nb_samples
        self.emb_size = emb_size
        self.sample = nb_samples #TODO to remove!
        self.tcr_input_size = tcr_input_size
        self.hla_input_size = hla_input_size

        layers = []
        outsize = self.tcr_input_size

        for i in range(0,len(tcr_conv_layers_sizes),3):
            layer = nn.Conv1d(in_channels = tcr_conv_layers_sizes[i+0],out_channels = tcr_conv_layers_sizes[i+1],kernel_size = tcr_conv_layers_sizes[i+2],stride = 1)
            outsize = outsize-tcr_conv_layers_sizes[i+2]+1
            layers.append(layer)
            dim1 = [(tcr_conv_layers_sizes[i+1]*(outsize))]

        self.tcr_conv_stack = nn.ModuleList(layers)
        dim1 = dim1[0]
        if not dim1==emb_size:
            import pdb;pdb.set_trace()

        layers = []
        outsize = self.hla_input_size

        for i in range(0,len(hla_conv_layers_sizes),3):
            layer = nn.Conv1d(in_channels = hla_conv_layers_sizes[i+0],out_channels = hla_conv_layers_sizes[i+1],kernel_size = hla_conv_layers_sizes[i+2],stride = 1)
            outsize = outsize-hla_conv_layers_sizes[i+2]+1
            layers.append(layer)
            dim2 = [(hla_conv_layers_sizes[i+1]*(outsize))]

        self.hla_conv_stack = nn.ModuleList(layers)
        dim2 = dim2[0]
        if not dim2==emb_size:
            import pdb; pdb.set_trace()
        self.hla_mlp = nn.Linear(4*dim2, emb_size)

        layers = []
        dim = [emb_size*2] + mlp_layers_size
        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(dim[-1], 1)

    def encode_hla(self, hla):

        for layer in self.hla_conv_stack:
            hla = layer(hla)
        return hla

    def encode_tcr(self,tcr):

        for layer in self.tcr_conv_stack:
            tcr = layer(tcr)

        return tcr


    def get_embeddings(self, x1, x2, x3, x4, x5):

        tcr, hla1, hla2, hla3, hla4 = x1, x2, x3, x4, x5
        tcr = self.encode_tcr(tcr)
        hla1 = self.encode_hla(hla1)
        hla2 = self.encode_hla(hla2)
        hla3 = self.encode_hla(hla3)
        hla4 = self.encode_hla(hla4)
        
        return tcr, hla1, hla2, hla3, hla4


    def get_hla_rep(self, h1, h2, h3, h4): ### TODO: this should be called get individual rep.
        self.hla_representation = torch.cat([h1.permute(1,0), h2.permute(1,0),
                                             h3.permute(1,0), h4.permute(1,0)])
        self.hla_representation = self.hla_mlp(self.hla_representation.permute(1,0))
        self.hla_representation = F.tanh(self.hla_representation)
        return self.hla_representation

    def forward(self, x1, x2, x3, x4, x5):

        # Get the embeddings
        emb_1, emb_2, emb_3, emb_4, emb_5 = self.get_embeddings(x1, x2, x3, x4, x5)
        emb_1 = emb_1.permute(1,0,2)
        emb_1 = emb_1.squeeze()
        emb_2 = emb_2.squeeze()
        emb_3 = emb_3.squeeze()
        emb_4 = emb_4.squeeze()
        emb_5 = emb_5.squeeze()

        if not emb_1.shape == emb_2.shape:
            import pdb; pdb.set_trace()
        hla_representation = self.get_hla_rep(emb_2, emb_3, emb_4, emb_5)
        mlp_input = torch.cat([emb_1, hla_representation], 1)
        # Forward pass.
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)
        mlp_output = self.last_layer(mlp_input)

        return mlp_output


class AllSeqCNNbin(nn.Module):

    def __init__(self, tcr_conv_layers_sizes = [20,1,18], 
        hla_conv_layers_sizes = [20,1,25], 
        mlp_layers_size = [25,10],
        tcr_input_size = 27,
        hla_input_size = 34, 
        nb_samples=1, emb_size=10, data_dir ='.'):
        super(AllSeqCNNbin, self).__init__()

        self.emb_size = emb_size
        self.sample = nb_samples #TODO to remove!
        self.tcr_input_size = tcr_input_size
        self.hla_input_size = hla_input_size

        layers = []
        outsize = self.tcr_input_size

        for i in range(0,len(tcr_conv_layers_sizes),3):
            layer = nn.Conv1d(in_channels = tcr_conv_layers_sizes[i+0],out_channels = tcr_conv_layers_sizes[i+1],kernel_size = tcr_conv_layers_sizes[i+2],stride = 1)
            outsize = outsize-tcr_conv_layers_sizes[i+2]+1
            layers.append(layer)
            dim1 = [(tcr_conv_layers_sizes[i+1]*(outsize))]

        self.tcr_conv_stack = nn.ModuleList(layers)
        dim1 = dim1[0]
        if not dim1==emb_size:
            import pdb; pdb.set_trace()

        layers = []
        outsize = self.hla_input_size

        for i in range(0,len(hla_conv_layers_sizes),3):
            layer = nn.Conv1d(in_channels = hla_conv_layers_sizes[i+0],out_channels = hla_conv_layers_sizes[i+1],kernel_size = hla_conv_layers_sizes[i+2],stride = 1)
            outsize = outsize-hla_conv_layers_sizes[i+2]+1
            layers.append(layer)
            dim2 = [(hla_conv_layers_sizes[i+1]*(outsize))]

        self.hla_conv_stack = nn.ModuleList(layers)
        dim2 = dim2[0]
        if not dim2==emb_size:
            import pdb;pdb.set_trace()
        self.hla_mlp = nn.Linear(4*dim2, emb_size)

        layers = []
        dim = [emb_size*2] + mlp_layers_size
        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)
        self.last_layer = nn.Linear(dim[-1], 1)
        self.softmax = nn.Softmax(dim=1)


    def encode_hla(self, hla):

        for layer in self.hla_conv_stack:
            hla = layer(hla)
        return hla

    def encode_tcr(self,tcr):

        for layer in self.tcr_conv_stack:
            tcr = layer(tcr)

        return tcr


    def get_embeddings(self, x1, x2, x3, x4, x5):

        tcr, hla1, hla2, hla3, hla4 = x1, x2, x3, x4, x5
        tcr = self.encode_tcr(tcr)
        hla1 = self.encode_hla(hla1)
        hla2 = self.encode_hla(hla2)
        hla3 = self.encode_hla(hla3)
        hla4 = self.encode_hla(hla4)
        
        return tcr, hla1, hla2, hla3, hla4


    def get_hla_rep(self, h1, h2, h3, h4): ### TODO: this should be called get individual rep.
        self.hla_representation = torch.cat([h1.permute(1,0), h2.permute(1,0),
                                             h3.permute(1,0), h4.permute(1,0)])
        self.hla_representation = self.hla_mlp(self.hla_representation.permute(1,0))
        self.hla_representation = F.tanh(self.hla_representation)
        return self.hla_representation

    def forward(self, x1, x2, x3, x4, x5):

        # Get the embeddings
        emb_1, emb_2, emb_3, emb_4, emb_5 = self.get_embeddings(x1, x2, x3, x4, x5)
        emb_1 = emb_1.permute(1,0,2)
        emb_1 = emb_1.squeeze()
        emb_2 = emb_2.squeeze()
        emb_3 = emb_3.squeeze()
        emb_4 = emb_4.squeeze()
        emb_5 = emb_5.squeeze()

        if not emb_1.shape == emb_2.shape:
            import pdb; pdb.set_trace()
        hla_representation = self.get_hla_rep(emb_2, emb_3, emb_4, emb_5)
        mlp_input = torch.cat([emb_1, hla_representation], 1)
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)
        mlp_output = self.last_layer(mlp_input)
        mlp_output = self.softmax(mlp_output)

        return mlp_output




def get_model(opt, inputs_size, model_state=None):

    if opt.model == 'TCRonly' or opt.model=='RNN':
        model_class = FactorizedCNN
        model = model_class(conv_layers_sizes = opt.conv_layers_sizes, 
            mlp_layers_size = opt.mlp_layers_size, nb_samples=inputs_size[0], emb_size=opt.emb_size, 
            tcr_input_size = opt.tcr_input_size, data_dir =opt.data_dir)

    elif opt.model=='allseq':
        model_class = AllSeqCNN
        model = model_class(tcr_conv_layers_sizes = opt.tcr_conv_layers_sizes, 
            hla_conv_layers_sizes = opt.hla_conv_layers_sizes, mlp_layers_size = opt.mlp_layers_size,
            tcr_input_size = opt.tcr_size, hla_input_size = opt.hla_size, 
            nb_samples=inputs_size[0], emb_size=opt.emb_size, data_dir =opt.data_dir)

    elif opt.model=='allseq_bin':
        model_class = AllSeqCNNbin
        model = model_class(tcr_conv_layers_sizes = opt.tcr_conv_layers_sizes, 
            hla_conv_layers_sizes = opt.hla_conv_layers_sizes, mlp_layers_size = opt.mlp_layers_size,
            tcr_input_size = opt.tcr_size, hla_input_size = opt.hla_size, 
            nb_samples=inputs_size[0], emb_size=opt.emb_size, data_dir =opt.data_dir)

    else:
        raise NotImplementedError()

    if model_state is not None:
        model.load_state_dict(model_state)

    return model
