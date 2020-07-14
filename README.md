# S2E
ICML'20: Searching to Exploit Memorization Effect in Learning from Corrupted Labels (PyTorch implementation).

This is the code for the paper: [Searching to Exploit Memorization Effect in Learning from Corrupted Labels](https://arxiv.org/abs/1911.02377)
Quanming Yao, Hansi Yang, Bo Han, Gang Niu, James Kwok.

```bash
@inproceedings{s2e2020icml,
  title={Searching to Exploit Memorization Effect in Learning from Corrupted Labels},
  author={Yao, Quanming and Yang, Hansi and Han, Bo and Niu, Gang and Kwok, James},
  booktitle={International Conference on Machine Learning},
  year={2020}
}
```

## Requirements
Python = 3.7, PyTorch = 1.3.1, NumPy = 1.18.5, SciPy = 1.4.1
All packages can be installed by Conda.

## Running S2E on benchmark datasets
Example usage for MNIST with 50% symmetric noise
```
python heng_mnist_main.py --noise_type symmetric --noise_rate 0.5 --num_workers 1 --n_iter 10 --n_samples 6
```

CIFAR-10 with 50% symmetric noise
```
python heng_main.py --noise_type symmetric --noise_rate 0.5 --num_workers 1 --n_iter 10 --n_samples 6
```

And CIFAR-100 with 50% symmetric noise
```
python heng_100_main.py --noise_type symmetric --noise_rate 0.5 --num_workers 1 --n_iter 10 --n_samples 6
```

Or see scripts (.sh files) for a quick start.

## Relavent resources
- A comprehensive survey on AutoML from our group is [here](http://xxx.itp.ac.cn/abs/1810.13306).
- Implementation of [Co-teaching](https://github.com/bhanML/Co-teaching) (the most important baseline in S2E).

## Example applications
S2E (AutoML version of Co-teaching) is based on the small-loss trick and the [memorization effect](https://arxiv.org/abs/1706.05394), the following examples have applied these princeples in various applications
- Zhang et.al. Collaborative Unsupervised Domain Adaptation for Medical Image Diagnosis. Medical Imaging meets NeurIPS 2019.
- Luo et.al. Deep Mining External Imperfect Data for Chest X-ray Disease Screening. IEEE Transactions on Medical Imaging. 2020.
- Yang et.al. Asymmetric Co-Teaching for Unsupervised Cross-Domain Person Re-Identification. AAAI 2020.
- Li et.al. Learning from Noisy Anchors for One-stage Object Detection. CVPR 2020.
- Wang et.al. Co-Mining: Deep Face Recognition with Noisy Labels. ICCV 2020.

## New Opportunities
- Interns, research assistants, and researcher positions are available. See [requirement](http://www.cse.ust.hk/~qyaoaa/pages/job-ad.pdf)
