from __future__ import print_function
from __future__ import division

import argparse

from classifier.config import get_cfg
from classifier.nets import build_model
from classifier.data import create_dataloaders
from classifier.engine import create_logger,train_classifier,test_classifier,run_classifier

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
    logger = create_logger(cfg)
    return cfg,logger

def main(args):
    cfg,logger = setup(args)
    model = build_model(cfg)
    dataloaders = create_dataloaders(cfg)
    logger.super_print(cfg.dump())
    model = train_classifier(model, dataloaders, logger, args.num_gpus, cfg)
    acc = test_classifier(model, dataloaders['test'], logger, args.num_gpus, cfg)
    #run_classifier(model, dataloaders['inference'], logger, args.num_gpus, cfg)

if __name__ == "__main__":
    args = return_parser().parse_args()
    main(args)
