import os
import torch
import argparse
import os.path as osp
import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.3.2 ", config_path = ".", config_name="simclr_config")
def test(args: DictConfig) -> None:
    # load the data
    folder = osp.join(osp.dirname(args.collapse_pretrained))
    data = torch.load(args.collapse_pretrained)
    reprs = data['representations']
    reprs = reprs.reshape(-1, reprs.shape[-1])

    norms = torch.linalg.norm(reprs, dim=1)
    if args.normalize:
        normed_reprs = reprs / (1e-6 + norms.unsqueeze(1))
    else:
        normed_reprs = reprs
    normed_reprs -= normed_reprs.mean(dim=0, keepdims=True)
    if args.type == 'svd':
        stds = torch.svd(normed_reprs).S
    elif args.type == 'std':
        stds = torch.std(normed_reprs, dim=0)
    else:
        raise NotImplementedError

    # save norms and std
    normalize_str = 'normalized_' if args.normalize else ''
    fname = f"{os.path.basename(args.collapse_pretrained).split('.')[0]}_{normalize_str}{args.type}.pt"
    # torch.save(dict(norms=norms, stds=stds), osp.join(folder, fname))
    torch.save(dict(stds=stds), osp.join(folder, fname))

if __name__ == '__main__':
    test()



