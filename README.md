# SimCLR-nashAE
Experiments done with integrating the SimCLR contrastive learning model with the nashAE frameowrk for disentangling the feature space.

# Setup

Hydra is a python framework to manage the hyperparameters during training and evaluation. Install with:

    conda install -c conda-forge hydra
### Dependencies:

- pytorch >=1.2
- torchvision >=0.4.0
- hydra >=0.11.3
- tqdm >=4.45.0



# Training
### SimCLR pre-training

Train SimCLR with resnet18 as backbone:

    python simclr.py backbone=resnet18

# Computing Metrics
### Linear Evaluation:

    python simclr_lin.py backbone=resnet18
### Collapse:


The default batch_size is 512. All the hyperparameters are available in simclr_config.yaml, which could be overrided from the command line.

# Evaluating 
