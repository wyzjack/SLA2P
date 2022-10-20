import argparse
import os
from datetime import datetime
from multiprocessing import Manager
import numpy as np
import types
from utils import save_roc_pr_curve_data, get_class_name_from_index
from outlier_datasets import load_cifar10_with_outliers, load_cifar100_with_outliers,load_20news_with_outliers, load_reuters_with_outliers, load_caltech_with_outliers
from models.fcn_pytorch import fcn
from keras2pytorch_dataset import trainset_pytorch, testset_pytorch
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from misc import AverageMeter
from eval_accuracy import simple_accuracy
from PIL import Image
from data_loader import Data_Loader
from reproduce import initialize
from evaluate_roc_auc import *
from evaluate_pr_auc import *

parser = argparse.ArgumentParser(description='Run UAD experiments.')
parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save results.')

parser.add_argument('--n_rots', type=int, default= 256)
parser.add_argument('--n_run', type=int, default= 5)
parser.add_argument('--d_out', type=int, default= 256)
parser.add_argument('--acc_thres', type=float, default= 0.6)
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'caltech', 'reuters', '20news', 'arrhythmia', 'kdd'],
                    help='dataset name for UAD')
parser.add_argument('--extract_model', type=str, default='res50',
                    choices=['res50', 'res101'],
                    help='pretrained model for feature extracting')
parser.add_argument('--epsilon', type=float, default= 1000)

args = parser.parse_args()
if args.dataset in ['cifar10', 'cifar100', 'caltech']:
    RESULTS_DIR = args.results_dir + '_nrots_' + str(args.n_rots) + '_dout_' + str(args.d_out) + '_thres_' + str(args.acc_thres) + '_epsilon_' + str(args.epsilon) + '_extract_' + str(args.extract_model)
else:
    RESULTS_DIR = args.results_dir + '_nrots_' + str(args.n_rots) + '_dout_' + str(args.d_out) + '_thres_' + str(args.acc_thres) + '_epsilon_' + str(args.epsilon) 






def softmax(input_tensor):
    act = nn.Softmax(dim=1)
    return act(input_tensor).numpy()

def neg_entropy(score):
    if len(score.shape) != 1:
        score = np.squeeze(score)
    return score@np.log2(score+1e-16)

def dist_calc(feats1, feats2):
    nb_data1 = feats1.shape[0]
    nb_data2 = feats2.shape[0]
    omega = np.dot(np.sum(feats1 ** 2, axis=1)[:, np.newaxis], np.ones(shape=(1, nb_data2)))
    omega += np.dot(np.sum(feats2 ** 2, axis=1)[:, np.newaxis], np.ones(shape=(1, nb_data1))).T
    omega -= 2 * np.dot(feats1, feats2.T)
    return omega


def l2_loss(score,y):
    if len(score.shape) != 1:
        score = np.squeeze(score)
    if len(y) != 1:
        y = np.squeeze(y)
    return ((score - y)**2).mean(axis=-1)


def train_self_supervised(trainloader, model, criterion, optimizer, epochs):
    # train the model
    model.train()
    model.cuda()
    top1 = AverageMeter()
    losses = AverageMeter()
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = torch.autograd.Variable(inputs.float().cuda()),torch.autograd.Variable(targets.cuda())

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            prec1 = simple_accuracy(outputs.data.cpu(), targets.data.cpu())

            top1.update(prec1, inputs.size(0))
            losses.update(loss.data.cpu(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('Epoch: [{} | {}], batch: {}, loss: {}, Accuracy: {}'.format(epoch + 1, epochs, batch_idx + 1, losses.avg, top1.avg))

            if top1.avg > args.acc_thres:
                break

        if top1.avg > args.acc_thres:
            break

def test_self_supervised_perturb(testloader, model, criterion, noisemagnitude):
    res = torch.Tensor()
    model.eval()
    for batch_idx, (inputs) in enumerate(testloader):
        inputs= torch.autograd.Variable(inputs.float().cuda(), requires_grad= True)
        outputs = model(inputs)
        model_outputs = outputs.data.cpu().numpy()
        model_outputs = model_outputs - np.max(model_outputs, axis=1)[:, np.newaxis]
        model_outputs = np.exp(model_outputs) / np.sum(np.exp(model_outputs), axis=1)[:, np.newaxis]

        maxIndex = np.argmax(model_outputs, axis=1)

        labels = torch.autograd.Variable(torch.LongTensor(maxIndex).cuda())
        loss = criterion(outputs, labels)
        loss.backward()
        gradient = inputs.grad.data
 

        inputs = torch.add(inputs.data, noisemagnitude, gradient)
        # inputs_temp = torch.add(inputs_requires.data, 0, torch.ones_like(inputs_requires))
        inputs = torch.autograd.Variable(inputs.float().cuda())

        
        outputs = model(inputs)
        
        res = torch.cat((res, outputs.data.cpu()), dim=0)
    return res

def _forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    # x = self.fc(x)

    return x

def sla2p_experiment(args, x_train, y_train, dataset_name, single_class_ind, gpu_q, p):
    gpu_to_use = gpu_q.get()
    batch_size = 128
    cudnn.benchmark = True
    if dataset_name in ["cifar10", "cifar100", "caltech"]:
        epochs = 1
        if args.extract_model == 'res50':
            feature_extractor = models.resnet50(pretrained=True)
        elif args.extract_model == 'res101':
            feature_extractor = models.resnet101(pretrained=True)
        else:
            raise NotImplementedError
        feature_extractor.eval()
        feature_extractor.forward = types.MethodType(_forward, feature_extractor)
        feature_extractor.cuda()
        x_train_pil = []
        for i in range(x_train.shape[0]):
            x_train_pil.append (Image.fromarray(x_train[i]))
        batch_size_extract = 10
        x_train_task=np.zeros((x_train.shape[0], 2048))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.25])
        dataset_pytorch = trainset_pytorch(train_data=x_train_pil, train_labels=y_train,
                                        transform=transforms.Compose([transforms.Resize((224,224)),
                                                                                                    transforms.ToTensor(),
                                                                                                    normalize]),
                                        )
        feature_loader =data.DataLoader(dataset_pytorch, batch_size=batch_size_extract, shuffle = False)
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(feature_loader):
                if inputs.shape[0]!= batch_size_extract:
                    out = feature_extractor(inputs.cuda())
                    x_train_task[batch_idx*batch_size_extract:batch_idx*batch_size_extract+inputs.shape[0]]=out.cpu().data.numpy()
                else:
                    out = feature_extractor(inputs.cuda())
                    x_train_task[batch_idx*batch_size_extract: (batch_idx+1)*batch_size_extract] = out.cpu().data.numpy()
        for i in range(x_train_task.shape[0]):
            x_train_task[i] = (x_train_task[i] / np.linalg.norm(x_train_task[i]))
    elif dataset_name in ['reuters', '20news']:
        epochs=1000000
        x_train_task = x_train
    elif dataset_name in ['kdd', 'arrhythmia']:
        epochs=1000000
        x_train_task = x_train
        x_train_task = x_train_task / np.linalg.norm(x_train_task, axis=1)[:,np.newaxis]

    # self-supervised learning
    n_train, n_dims = x_train_task.shape
  
    rots = np.random.randn(args.n_rots, n_dims, args.d_out)

    print('Calculating transforms')

    x_train_task = np.stack([x_train_task.dot(rot) for rot in rots], 1).reshape((-1,args.d_out))
    transformations_label = np.tile(np.arange(args.n_rots), len(x_train))

    trainset_self = trainset_pytorch(train_data=x_train_task, train_labels=transformations_label)
    trainloader_self = data.DataLoader(trainset_self, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    #classifier-model
    model = fcn(in_features_num=args.d_out, class_num=args.n_rots)
    #optimize and train
    if dataset_name == 'reuters':
        optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-8, weight_decay=0)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-7, weight_decay=0.0005)
    train_self_supervised(trainloader_self, model, criterion, optimizer, epochs)
 
    preds = np.zeros((np.shape(x_train)[0], args.n_rots), dtype='float32')
    original_preds = np.zeros((args.n_rots, np.shape(x_train)[0], args.n_rots), dtype='float32')

    for t in range(args.n_rots):
        y_l = np.zeros(args.n_rots)
        y_l[t] = 1
        idx = np.squeeze(np.array([range(x_train.shape[0])]) * args.n_rots + t)
        test_set = testset_pytorch(test_data=x_train_task[idx, :])
        original_preds[t, :, :] = softmax(
            test_self_supervised_perturb(testloader=data.DataLoader(test_set, batch_size=batch_size, shuffle=False), model=model, criterion=criterion, noisemagnitude=args.epsilon))
        for s in range(np.shape(x_train)[0]):
            preds[s, t] = -l2_loss(original_preds[t, s, :], y_l)
    scores = preds.mean(axis=-1)

    # save
    if args.dataset in ['kdd', 'arrhythmia']:
        res_file_name = '{}_sla2p_{}.npz'.format(dataset_name, datetime.now().strftime('%Y-%m-%d-%H%M%S'))
    else:
        res_file_name = '{}_sla2p-outlier_{}_{}_{}.npz'.format(dataset_name, p,  
                                                            get_class_name_from_index(single_class_ind, dataset_name),
                                                            datetime.now().strftime('%Y-%m-%d-%H%M%S'))
    res_file_path = os.path.join(RESULTS_DIR, dataset_name, res_file_name)
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)
    save_roc_pr_curve_data(scores, y_train, res_file_path)
    gpu_q.put(gpu_to_use)



# ############################### Interface to run all experiments ###################################################

def run_experiments(load_dataset_fn, dataset_name, q, n_classes, abnormal_fraction, run_idx):
    # reproducibility 
    initialize(run_idx)

    max_sample_num = 12000
    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)

    for c in range(n_classes):
        x_train, y_train = load_dataset_fn(c, abnormal_fraction)
        # random sampling if the number of data is too large
        if x_train.shape[0] > max_sample_num:
            selected = np.random.choice(x_train.shape[0], max_sample_num, replace=False)
            x_train = x_train[selected, :]
            y_train = y_train[selected]
        sla2p_experiment(args, x_train, y_train, dataset_name, c, q, abnormal_fraction)



def run_experiments_intrinsic(dataset_name, q, run_idx):
    # reproducibility 
    initialize(run_idx)

    os.makedirs(os.path.join(RESULTS_DIR, dataset_name), exist_ok=True)
    dl = Data_Loader()
    x_train, y_train = dl.get_dataset(dataset_name)
    sla2p_experiment(args, x_train, y_train, dataset_name, 0, q, 0.1)


if __name__ == '__main__':
    # hard
    N_GPUS = 1  # deprecated, use one gpu only
    man = Manager()
    q = man.Queue(N_GPUS)
    for g in range(N_GPUS):
        q.put(str(g))

    if args.dataset in ['cifar10', 'cifar100', 'caltech', 'reuters', '20news']:
        if args.dataset == 'cifar10':
            data_load_fn = load_cifar10_with_outliers
            n_classes = 10
        elif args.dataset == 'cifar100':
            data_load_fn = load_cifar100_with_outliers
            n_classes = 20
        elif args.dataset == 'caltech':
            data_load_fn = load_caltech_with_outliers
            n_classes = 11
        elif args.dataset == '20news':
            data_load_fn = load_20news_with_outliers
            n_classes = 20
        elif args.dataset == 'reuters':
            data_load_fn = load_reuters_with_outliers
            n_classes = 5
        
        p_list = [0.1, 0.3, 0.5, 0.01, 0.02, 0.03, 0.04, 0.05]
        for p in p_list:
            for i in range(args.n_run):
                run_experiments(data_load_fn, args.dataset, q, n_classes, p, i)
            # test AUROC and AUPR

            algo_name = 'sla2p-outlier_' + str(p)
            n_classes = {
                'cifar10': 10, 'mnist': 10, 'cifar100': 20, 'fashion-mnist': 10, 'svhn': 10, '20news': 20, 'reuters': 5,
                'caltech': 11
            }[args.dataset]
            compute_average_roc_auc_traintest(algo_name, RESULTS_DIR, args.dataset,
                                n_classes, p)
            compute_average_pr_auc_traintest(p, algo_name, RESULTS_DIR, args.dataset,
                                n_classes, positive='outliers')
    elif args.dataset in ['arrhythmia', 'kdd']:
        for i in range(args.n_run):
            run_experiments_intrinsic(args.dataset, q, i)
        algo_name = 'sla2p'
        compute_roc_auc_traintest(algo_name, RESULTS_DIR, args.dataset)
        compute_pr_auc_traintest(algo_name, RESULTS_DIR, args.dataset, positive='outliers')
    else:
        raise NotImplementedError
        
