# SLA$^2$P: Self-supervised Anomaly Detection with Adversarial Perturbation （CIKM 2022）


<div align="left">
    <a><img src="images/smile.png"  height="70px" ></a>
    <a><img src="images/neu.png"  height="70px" ></a>
    <a><img src="images/purdue_smaller.png"  height="70px" ></a>
</div>

Primary contact: [Yizhou Wang](mailto:wyzjack990122@gmail.com)

<div align="center">
  <img src="images/framework.png" width="850px" height="250px">
</div>


## Abstract

Anomaly detection is a fundamental yet challenging problem in machine learning due to the lack of label information. In this work, we propose a novel and powerful framework, dubbed as SLA$^2$P, for unsupervised anomaly detection. After extracting representative embeddings from raw data, we apply random projections to the features and regard features transformed by different projections as belonging to distinct pseudo-classes. We then train a classifier network on these transformed features to perform self-supervised learning. Next, we add adversarial perturbation to the transformed features to decrease their softmax scores of the predicted labels and design anomaly scores based on the predictive uncertainties of the classifier on these perturbed features. Our motivation is that because of the relatively small number and the decentralized modes of anomalies, **1)** the pseudo label classifier's training concentrates more on learning the semantic information of normal data rather than anomalous data; **2)** the transformed features of the normal data are more robust to the perturbations than those of the anomalies. Consequently, the perturbed transformed features of anomalies fail to be classified well and accordingly have lower anomaly scores than those of the normal samples. Extensive experiments on image, text, and inherently tabular benchmark datasets back up our findings and indicate that SLA$^2$P achieves state-of-the-art anomaly detection performance consistently.

## TODO
The baseline codes will be added soon.

## Usage

#### Prepare data

Download processed data (Caltech 101, 20 Newsgroups and Reuters) from [[Google Drive Link]](https://drive.google.com/drive/folders/11Bvi9x3dfDql5tov4NmEgVtLGQRFKPSy?usp=sharing) and put them into folder `/data`. They are borrowed from the [[official implementation of RSRAE]](https://github.com/dmzou/RSRAE) (Robust Subspace Recovery Layer for Unsupervised Anomaly Detection. ICLR 2020)


#### Environment setup

```bash
conda env create -f env.yaml
```

#### Run the experiments

The SLA$^2$P method is implemented in `sla2p.py` and the SLA (w/o adversarial perturbation) method is in `sla.py`. 

To reproduce the results reported in the main paper, run the following commands.

```bash
# CIFAR-10
python sla2p.py --dataset cifar10 --n_rots 256 --d_out 256 --acc_thres 0.6 --epsilon 1000

# CIFAR-100
python sla2p.py --dataset cifar100 --n_rots 256 --d_out 256 --acc_thres 0.6 --epsilon 10000

# Caltech 101
python sla2p.py --dataset caltech --n_rots 256 --d_out 256 --acc_thres 0.6 --epsilon 1000

# 20 Newsgroups
python sla2p.py --dataset 20news --n_rots 256 --d_out 256 --acc_thres 0.75 --epsilon 10

# Reuters
python sla2p.py --dataset reuters --n_rots 512 --d_out 128 --acc_thres 0.3 --epsilon 100

# Arrhythmia
python sla2p.py --dataset arrhythmia --n_rots 256 --d_out 256 --acc_thres 0.6 --epsilon 1000

# KDD
python sla2p.py --dataset kdd --n_rots 64 --d_out 128 --acc_thres 0.6 --epsilon 1000

``` 

To evalute Unsupervised Anomaly Detection performance, use `evaluate_roc_auc.py` for AUROC scores and `evaluate_pr_auc.py` for AUPR scores.



## Acknowledgments
In this code we heavily rely on the [code of E3Outlier](https://github.com/demonzyj56/E3Outlier). The README file format is heavily based on the GitHub repos of my senior colleague [Huan Wang](https://github.com/MingSun-Tse). Great thanks to them! We also greatly thank the anounymous CIKM'22 reviewers for the constructive comments to help us improve the paper. 


## BibTeX

```BibTeX
@inproceedings{wang2022self,
  title={Self-supervision Meets Adversarial Perturbation: A Novel Framework for Anomaly Detection},
  author={Wang, Yizhou and Qin, Can and Wei, Rongzhe and Xu, Yi and Bai, Yue and Fu, Yun},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={4555--4559},
  year={2022}
}
```
