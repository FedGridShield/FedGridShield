from model.cnn import CNNFmnist, CNNSvhn, CNNCifar, CNNdefectclassification
from model.mlp import MLP, MLPtimeseries
from model.recurrent import RNN_FedShakespeare
from model.resnet import ResNet9FashionMNIST, ResNet18, ReducedResNet18, \
                         CIFARResNet20, SVHNResNet20, ResNetTabular, ResNetDefectClassification
from model.ac_model import TransformerTabular
from model.lstm import ETD_Model
import torch
import copy

################################### model setup ########################################
def model_setup(args):
    if args.model == 'cnn_defectclassification':
        # net_glob = CNNdefectclassification().to(args.device)
        net_glob = ResNetDefectClassification().to(args.device)
        # for param in net_glob.resnet.parameters():
        #     param.requires_grad = False
    elif args.model == 'resnet_tabularclassification':
        net_glob = TransformerTabular().to(args.device)
    # elif args.model == 'mlp_timeseries':
    #     net_glob = MLPtimeseries().to(args.device)
    elif args.model == 'lstm_elec':
        net_glob = ETD_Model().to(args.device)
    else:
        exit('Error: unrecognized model')
    global_model = copy.deepcopy(net_glob.state_dict())
    return args, net_glob, global_model, model_dim(global_model)

def model_dim(model):
    '''
    compute model dimension
    '''
    flat = [torch.flatten(model[k]) for k in model.keys()]
    s = 0
    for p in flat: 
        s += p.shape[0]
    return s


def model_clip(model, clip):
    '''
    clip model update
    '''
    model_norm=[]
    for k in model.keys():
        if 'num_batches_tracked' in k or 'running_mean' in k or 'running_var' in k:
            continue
        model_norm.append(torch.norm(model[k]))
        
    total_norm = torch.norm(torch.stack(model_norm))
    clip_coef = clip / (total_norm + 1e-8)
    if clip_coef < 1:
        for k in model.keys():
            if 'num_batches_tracked' in k or 'running_mean' in k or 'running_var' in k:
                continue
            model[k] = model[k] * clip_coef
    return model, total_norm

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


def get_trainable_values(net,mydevice=None):
    ' return trainable parameter values as a vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable) 
    N=0
    for params in paramlist:
        N+=params.numel()
    if mydevice:
        X=torch.empty(N,dtype=torch.float).to(mydevice)
    else:
        X=torch.empty(N,dtype=torch.float)
    X.fill_(0.0)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            X[offset:offset+numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset+=numel

    return X

def put_trainable_values(net,X):
    ' replace trainable parameter values by the given vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset+=numel
