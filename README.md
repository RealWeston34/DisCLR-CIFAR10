# SimCLR-nashAE
Experiments done with integrating the SimCLR contrastive learning model with the nashAE frameowrk for disentangling the projection head.
<img width="920" alt="Screenshot 2023-08-13 163606" src="https://github.com/RealWeston34/DisCLR-CIFAR10/assets/73916480/7cbc1427-3fb8-486d-9d51-e192bc376102">

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
Accuracy is measured with a linear classifier finetuned for 100 epochs. Run:

    python simclr_lin.py backbone=resnet18
### Collapse:
Save representations by running the following command:

        python save_reprs.py

Save dictionary with svds by running: 

        python test_collapse.py

Test collapse metric with collapse_analysis notebook


