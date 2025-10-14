from __future__ import print_function
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def load_pretrained_model(model, pretrained_dict, wfc=True):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    if wfc:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((k in model_dict) and ('fc' not in k))}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def transform_time(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return h, m, s


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def defense_adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 10:
        lr = 0.01
    else:
        lr = 0.001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def fine_defense_adjust_learning_rate(optimizer, epoch, lr, dataset, mode = "default"):
    if dataset=='CIFAR10' or dataset=='imagenet' or dataset=='CIFAR100' or dataset == 'tinyImagenet':
        if mode == "default":
            if epoch < 1:
                lr = 0.001
            elif epoch < 10:
                lr = 0.001
            elif epoch < 20:
                lr = 0.0001
            else:
                lr = 0.0001
        elif mode == "CD":
            if epoch < 1:
                lr = 0.0001
            elif epoch < 10:
                lr = 0.001
            elif epoch < 20:
                lr = 0.001
            else:
                lr = 0.0001
    elif dataset=='gtsrb' or dataset == "ROF":
        if epoch < 2:
            lr = 0.001
        elif epoch < 10:
            lr = 0.0001
        elif epoch < 20:
            lr = 0.00001
        else:
            lr = 0.0001
    # elif dataset == 'ROF':
    #     lr = 0.0001
    else:
        raise Exception('Invalid dataset')
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 2:
        lr = lr
    elif epoch < 20:
        lr = 0.01
    elif epoch < 30:
        lr = 0.0001
    else:
        lr = 0.0001
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, fdir, model_name):
    filepath = os.path.join(fdir, model_name + '.tar')
    if is_best:
        torch.save(state, filepath)
        print('[info] save best model')


def save_history(cls_orig_acc, clease_trig_acc, cls_trig_loss, at_trig_loss, at_epoch_list, logs_dir):
    dataframe = pd.DataFrame({'epoch': at_epoch_list, 'cls_orig_acc': cls_orig_acc, 'clease_trig_acc': clease_trig_acc,
                              'cls_trig_loss': cls_trig_loss, 'at_trig_loss': at_trig_loss})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(logs_dir, index=False, sep=',')

def plot_curve(clean_acc, bad_acc, epochs, dataset_name):
    N = epochs+1
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), clean_acc, label="Classification Accuracy", marker='D', color='blue')
    plt.plot(np.arange(0, N), bad_acc, label="Attack Success Rate",  marker='o', color='red')
    plt.title(dataset_name)
    plt.xlabel("Epoch")
    plt.ylabel("Student Model Accuracy/Attack Success Rate(%)")
    plt.xticks(range(0, N, 1))
    plt.yticks(range(0, 101, 20))
    plt.legend()
    plt.show()

    '''
    ==> Preparing train data..
Files already downloaded and verified
full_train: 50000
train_size: 2500 drop_size: 47500
==> Preparing train data..
Files already downloaded and verified
full_train: 50000
train_size: 2500 drop_size: 47500
==> Preparing train data..
Files already downloaded and verified
full_train: 50000
train_size: 2500 drop_size: 47500
==> Preparing test data..

KEYS of PROTOS dict_keys(['layer4.1.conv2'])
PROTO SHAPE WITH NORMAL ACTS 0 (512, 4, 4)
PROTO SHAPE WITH NORMAL ACTS 1 (512, 4, 4)
PROTO SHAPE WITH NORMAL ACTS 2 (512, 4, 4)
PROTO SHAPE WITH NORMAL ACTS 3 (512, 4, 4)
PROTO SHAPE WITH NORMAL ACTS 4 (512, 4, 4)
PROTO SHAPE WITH NORMAL ACTS 5 (512, 4, 4)
PROTO SHAPE WITH NORMAL ACTS 6 (512, 4, 4)
PROTO SHAPE WITH NORMAL ACTS 7 (512, 4, 4)
PROTO SHAPE WITH NORMAL ACTS 8 (512, 4, 4)
PROTO SHAPE WITH NORMAL ACTS 9 (512, 4, 4)
new

PROTO SHAPE WITH DINO ACTS 0 (64, 4, 4)
PROTO SHAPE WITH DINO ACTS 1 (64, 4, 4)
PROTO SHAPE WITH DINO ACTS 2 (64, 4, 4)
PROTO SHAPE WITH DINO ACTS 3 (64, 4, 4)
PROTO SHAPE WITH DINO ACTS 4 (64, 4, 4)
PROTO SHAPE WITH DINO ACTS 5 (64, 4, 4)
PROTO SHAPE WITH DINO ACTS 6 (64, 4, 4)
PROTO SHAPE WITH DINO ACTS 7 (64, 4, 4)
PROTO SHAPE WITH DINO ACTS 8 (64, 4, 4)
PROTO SHAPE WITH DINO ACTS 9 (64, 4, 4)


Loading Model from ./weight/CIFAR10/preactresnet18-wanet.pth.tar
odict_keys(['conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias', 
'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var',
'layer1.0.bn1.num_batches_tracked', 'layer1.0.conv1.weight', 
'layer1.0.bn2.weight', 'layer1.0.bn2.bias', 'layer1.0.bn2.running_mean',
'layer1.0.bn2.running_var', 'layer1.0.bn2.num_batches_tracked',
'layer1.0.conv2.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias',
'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 
'layer1.1.bn1.num_batches_tracked', 'layer1.1.conv1.weight', 
'layer1.1.bn2.weight', 'layer1.1.bn2.bias', 'layer1.1.bn2.running_mean',
'layer1.1.bn2.running_var', 'layer1.1.bn2.num_batches_tracked', 
'layer1.1.conv2.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias', 
'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 
'layer2.0.bn1.num_batches_tracked', 'layer2.0.conv1.weight', 
'layer2.0.bn2.weight', 'layer2.0.bn2.bias', 'layer2.0.bn2.running_mean', 
'layer2.0.bn2.running_var', 'layer2.0.bn2.num_batches_tracked', 
'layer2.0.conv2.weight', 'layer2.0.shortcut.0.weight',
'layer2.1.bn1.weight', 'layer2.1.bn1.bias', 'layer2.1.bn1.running_mean',
'layer2.1.bn1.running_var', 'layer2.1.bn1.num_batches_tracked',
'layer2.1.conv1.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias', 
'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 
'layer2.1.bn2.num_batches_tracked', 'layer2.1.conv2.weight', 
'layer3.0.bn1.weight', 'layer3.0.bn1.bias', 'layer3.0.bn1.running_mean', 
'layer3.0.bn1.running_var', 'layer3.0.bn1.num_batches_tracked', 
'layer3.0.conv1.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias', 
'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 
'layer3.0.bn2.num_batches_tracked', 'layer3.0.conv2.weight', 
'layer3.0.shortcut.0.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias', 
'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 
'layer3.1.bn1.num_batches_tracked', 'layer3.1.conv1.weight', 
'layer3.1.bn2.weight', 'layer3.1.bn2.bias', 'layer3.1.bn2.running_mean', 
'layer3.1.bn2.running_var', 'layer3.1.bn2.num_batches_tracked', 
'layer3.1.conv2.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias', 
'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.num_batches_tracked', 
'layer4.0.conv1.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias', 
'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 
'layer4.0.bn2.num_batches_tracked', 'layer4.0.conv2.weight', 
'layer4.0.shortcut.0.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias', 'layer4.1.bn1.running_mean', 
'layer4.1.bn1.running_var', 'layer4.1.bn1.num_batches_tracked', 'layer4.1.conv1.weight', 'layer4.1.bn2.weight', 
'layer4.1.bn2.bias', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.num_batches_tracked', 
'layer4.1.conv2.weight', 'linear.weight', 'linear.bias'])


    '''