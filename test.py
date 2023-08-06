"""Given some pretrained model, get representations for every image in a desired dataset"""
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import torchvision
import torchvision.models as torchvision_models
from torchvision import datasets, transforms
from tqdm import tqdm

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='Saving MoCo representations')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture')
parser.add_argument('--dataset_type', default='cifar10_train', type=str)
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simclr pretrained checkpoint')
parser.add_argument('--pretrained_distributed', action='store_true',
                    help='was the pretrained checkpoint distributed?')
parser.add_argument('--exp_id', default=-1, type=int)
parser.add_argument('--extra_info', default=None, type=str, help='extra info to put in folder name')
# parser.add_argument('--alg', type=str, default='simsiam')


class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.enc = base_encoder(pretrained=False)  # load model from torchvision.models without pretrained weights.
        self.feature_dim = self.enc.fc.in_features

        # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
        # See Section B.9 of SimCLR paper.
        self.enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.enc.maxpool = nn.Identity()
        self.enc.fc = nn.Identity()  # remove final fully connected layer.

        # Add MLP projection.
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, projection_dim))

    def forward(self, x):
        feature = self.enc(x)
        projection = self.projector(feature)
        return feature, projection
    
def create_and_load_model(args):
    # create model
    print("=> creating model '{}'".format(args.arch))
    contrast_model = SimCLR(base_model=torchvision_models.__dict__[args.arch]())
    args.repr_size = contrast_model.feature_dim

    # load from checkpoint
    assert args.pretrained
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        contrast_model.load_state_dict(torch.load(args.pretrained))
        args.start_epoch = 0
        print("=> loaded pre-trained model '{}'".format(args.pretrained))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.pretrained))
    model = contrast_model.enc
    return model.cuda(args.gpu)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    print("Use GPU: {} for training".format(args.gpu))

    model = create_and_load_model(args)
    cudnn.benchmark = True

    # Load the dataset
    if args.dataset_type == 'cifar10_train':
        data_dir = os.path.abspath(args.data)
        val_trans = transforms.Compose([transforms.ToTensor(),])
        dataset = torchvision.datasets.CIFAR10(root=data_dir, 
                                               train=True, 
                                               transform=val_trans, 
                                               download=False)
    else:
        print("=> Your dataset is currently not supported.")
        exit()
    print("Dataset size:", len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True, drop_last=False)
    dataset_name, split = args.dataset_type.split('_')
    folder = f"data/{args.arch}_exp{args.exp_id:04d}"
    if args.extra_info:
        folder += f"_{args.extra_info}"
    folder += f"/{dataset_name}"

    os.makedirs(folder, exist_ok=True)
    for epoch in range(args.epochs):
        print('Computing representations')
        result = get_reprs(loader, model, args, len(dataset))
        fname = f"{split}_reprs_e{epoch}.pt"
        print('Saving representations')
        torch.save(dict(representations=result['representations'],
                        targets=result['targets']),
                   os.path.join(folder, fname))
        del result  # clearing memory


def get_reprs(train_loader, model, args, dataset_len):
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    representations = torch.zeros(dataset_len, args.repr_size, dtype=torch.float32, device='cpu')
    targets = torch.zeros(dataset_len)

    with torch.inference_mode():
        for i, (images, target) in enumerate(tqdm(train_loader)):
            images = images.cuda(args.gpu, non_blocking=True)
            output = model(images)
            representations[i * args.batch_size: i * args.batch_size + len(output)] = output.detach().cpu()
            targets[i * args.batch_size: i * args.batch_size + len(output)] = target.detach()

    return dict(representations=representations, targets=targets)


if __name__ == '__main__':
    main()

