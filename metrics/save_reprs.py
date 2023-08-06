"""Given some pretrained model, get representations for every image in a desired dataset"""
import hydra
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
from torchvision.models import resnet18, resnet34
from torchvision import datasets, transforms
from tqdm import tqdm
from omegaconf import DictConfig
from models import SimCLR

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

    
def create_and_load_model(args):
    # create model
    print("=> creating model '{}'".format(args.backbone))
    base_encoder = eval(args.backbone)
    contrast_model = SimCLR(base_encoder, projection_dim = args.projection_dim).cuda()
    args.repr_size = contrast_model.feature_dim

    # load from checkpoint
    assert args.reprs_pretrained
    if os.path.isfile(args.reprs_pretrained):
        print("=> loading checkpoint '{}'".format(args.reprs_pretrained))
        contrast_model.load_state_dict(torch.load(args.reprs_pretrained).state_dict())
        args.start_epoch = 0
        print("=> loaded pre-trained model '{}'".format(args.reprs_pretrained))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.reprs_pretrained))
    model = contrast_model.enc
    return model.cuda(args.gpu)



@hydra.main(version_base="1.3.2 ", config_path = ".", config_name="simclr_config")
def main(args: DictConfig) -> None:
    
    if args.reprs_seed is not None:
        random.seed(args.reprs_seed)
        torch.manual_seed(args.reprs_seed)
        np.random.seed(args.reprs_seed)

    print("Use GPU: {} for training".format(args.gpu))

    model = create_and_load_model(args)
    cudnn.benchmark = True

    # Load the dataset
    if args.reprs_dataset == 'cifar10_train':
        data_dir = os.path.abspath(args.reprs_data)
        val_trans = transforms.Compose([transforms.ToTensor(),])
        dataset = torchvision.datasets.CIFAR10(root=data_dir, 
                                               train=True, 
                                               transform=val_trans, 
                                               download=True)
    else:
        print("=> Your dataset is currently not supported.")
        exit()
    print("Dataset size:", len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.reprs_batch_size, shuffle=False,
                                         num_workers=args.reprs_workers, pin_memory=True, drop_last=False)
    dataset_name, split = args.reprs_dataset.split('_')
    folder = f"data/{args.backbone}_exp{args.exp_id:04d}"
    if args.extra_info:
        folder += f"_{args.extra_info}"
    folder += f"/{dataset_name}"

    os.makedirs(folder, exist_ok=True)
    for epoch in range(args.reprs_epochs):
        print('Computing representations')
        result = get_reprs(loader, model, args, len(dataset))
        fname = f"{split}_reprs_e{epoch}_{args.projection_dim}.pt"
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