import argparse
import os
import re
import numpy as np
from utils import get_class_name_from_index


def get_filenames(algo_name, results_dir, dataset_name, class_name):
    """Returns all files satisfying the patterns."""
    pattern = re.compile(r'{}_{}_{}_[0-9\-]+\.npz'.format(
        dataset_name, algo_name, class_name
    ))
    all_files = os.listdir(os.path.join(results_dir, dataset_name))
    selected = [f for f in all_files if pattern.match(f) is not None]
    return sorted([os.path.join(results_dir, dataset_name, f) for f in selected])

def get_filenames_no_class(algo_name, results_dir, dataset_name):
    """Returns all files satisfying the patterns."""
    pattern = re.compile(r'{}_{}_[0-9\-]+\.npz'.format(
        dataset_name, algo_name
    ))
    # print(pattern)
    all_files = os.listdir(os.path.join(results_dir, dataset_name))
    selected = [f for f in all_files if pattern.match(f) is not None]
    return sorted([os.path.join(results_dir, dataset_name, f) for f in selected])


def compute_average_pr_auc(algo_name, results_dir, dataset_name, n_classes, positive='normal'):
    results = {}
    avg_results = []
    std_results = []
    if positive == 'inliers':
        target = 'pr_auc_norm'
    elif positive == 'outliers':
        target = 'pr_auc_anom'
    else:
        raise KeyError(positive)
    for c in range(n_classes):
        class_name = get_class_name_from_index(c, dataset_name)
        filenames = get_filenames(algo_name, results_dir, dataset_name, class_name)
        results[class_name] = [np.load(f)[target] for f in filenames]
    for k, v in results.items():
        print('{}: {:.2f} +- {:.2f}'.format(k, 100*np.mean(v), 100*np.std(v)))
        avg_results.append(np.mean(v))
        std_results.append(np.std(v))
    # compute the std of average results over multiple runs
    min_runs = min([len(v) for v in results.values()])
    std_rec = []
    for i in range(min_runs):
        ith_run = [results[get_class_name_from_index(c, dataset_name)][i] for c in range(n_classes)]
        std_rec.append(np.mean(ith_run))
    print('-------------------------------------------')
    print('Average: {:.2f} +- {:.2f}'.format(100*np.mean(avg_results), 100*np.std(std_rec)))
 

def compute_average_pr_auc_traintest(p, algo_name, results_dir, dataset_name, n_classes, positive='normal'):
    results = {}
    avg_results = []
    std_results = []
    with open(os.path.join(results_dir, dataset_name)+'/aupr_{}.txt'.format(p), 'a') as txt_file:
            print(algo_name, file = txt_file)
    if positive == 'inliers':
        target = 'pr_auc_norm'
    elif positive == 'outliers':
        target = 'pr_auc_anom'
    else:
        raise KeyError(positive)
    for c in range(n_classes):
        class_name = get_class_name_from_index(c, dataset_name)
        filenames = get_filenames(algo_name, results_dir, dataset_name, class_name)
        results[class_name] = [np.load(f)[target] for f in filenames]
    for k, v in results.items():
        with open(os.path.join(results_dir, dataset_name)+'/aupr_{}.txt'.format(p), 'a') as txt_file:
            print(f'{k}: {100*np.mean(v):.2f} +- {100*np.std(v):.2f}', file = txt_file)
        avg_results.append(np.mean(v))
        std_results.append(np.std(v))
    # compute the std of average results over multiple runs
    min_runs = min([len(v) for v in results.values()])
    std_rec = []
    for i in range(min_runs):
        ith_run = [results[get_class_name_from_index(c, dataset_name)][i] for c in range(n_classes)]
        std_rec.append(np.mean(ith_run))
    with open(os.path.join(results_dir, dataset_name)+'/aupr_{}.txt'.format(p), 'a') as txt_file:
        print('-------------------------------------------', file = txt_file)

    

    with open(os.path.join(results_dir, dataset_name)+'/aupr_{}.txt'.format(p), 'a') as txt_file:
        print(f"AUPR: {100*np.mean(avg_results):.2f} +- {100*np.std(std_rec):.2f}", file = txt_file)


def compute_pr_auc(algo_name, results_dir, dataset_name, positive='normal'):

    if positive == 'inliers':
        target = 'pr_auc_norm'
    elif positive == 'outliers':
        target = 'pr_auc_anom'
    else:
        raise KeyError(positive)
    filenames = get_filenames_no_class(algo_name, results_dir, dataset_name)
    # print(filenames)
    results= [np.load(f)[target] for f in filenames]
    # print(results)

    print('Average: {:.2f} +- {:.2f}'.format(100*np.mean(results), 100*np.std(results)))

def compute_pr_auc_traintest(algo_name, results_dir, dataset_name, positive='normal'):

    if positive == 'inliers':
        target = 'pr_auc_norm'
    elif positive == 'outliers':
        target = 'pr_auc_anom'
    else:
        raise KeyError(positive)
    filenames = get_filenames_no_class(algo_name, results_dir, dataset_name)
    # print(filenames)
    results= [np.load(f)[target] for f in filenames]
    # print(results)
    print('Average: {:.2f} +- {:.2f}'.format(100*np.mean(results), 100*np.std(results)))

    with open(os.path.join(results_dir, dataset_name)+'/aupr.txt', 'a') as txt_file:
            print(algo_name, file = txt_file)

    with open(os.path.join(results_dir, dataset_name)+'/aupr.txt', 'a') as txt_file:
        print(f"AUPR: {100*np.mean(results):.2f} +- {100*np.std(results):.2f}", file = txt_file)



def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Argument parser for AUPR-in/AUPR-out evaluations.')
    parser.add_argument('--algo_name', type=str, default='sla2p-outlier-0.1')
    parser.add_argument('--positive', type=str, default='outliers', choices=['inliers', 'outliers'], help='Whether the positive class is inliers or outliers')
    parser.add_argument('--results_dir', type=str, default='./results_nrots_256_dout_256_thres_0.6_epsilon_1000.0_extract_res50')
    parser.add_argument('--dataset', type=str, default='cifar10')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == 'thyroid' or args.dataset == 'arrhythmia' or args.dataset == 'kdd':
        compute_pr_auc(args.algo_name, args.results_dir, args.dataset, args.positive)
    else:
        n_classes = {
            'cifar10': 10, 'mnist': 10, 'cifar100': 20, 'fashion-mnist': 10, 'svhn': 10, '20news': 20, 'caltech': 11, 'reuters':5,
        }[args.dataset]
        compute_average_pr_auc(args.algo_name, args.results_dir, args.dataset,
                               n_classes, args.positive)
