from __future__ import print_function
from __future__ import division

import argparse

from classifier.config import get_cfg
from classifier.nets import build_model
from classifier.data import create_dataloaders

def return_parser():
    parser = argparse.ArgumentParser(description = "Simple Classification.")

    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpu's to use")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="experimental settings")

    return parser

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    #default_setup(cfg, args)
    print(cfg)
    return cfg

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    dataloaders = create_dataloaders(cfg)
    for key in dataloaders:
        print(key)
        print(dataloaders[key])

if __name__ == "__main__":
    args = return_parser().parse_args()
    main(args)