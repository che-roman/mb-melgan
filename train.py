import torch
from torchaudio.functional import resample
import argparse
import yaml
import sys
import os
import time
from context import Context
from logger import Logger
from data import train_loader
from pqmf import design


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", "-l", required=True, type=str,
                        help="path to the folder for checkpoints and samples")
    parser.add_argument("--config", "-c", default=None,  type=str,
                        help="path to the config-file (if continuing training, should be None)")
    parser.add_argument("--iters", "-i", default=50000, type=int,
                        help="number of training iterations to be performed")
    parser.add_argument("--saving", "-s", default=5000, type=int,
                        help="interval between checkpoints (in iterations)")
    parser.add_argument("--tracking", "-t", default=100, type=int,
                        help="loss tracking interval (in iterations)")
    args = parser.parse_args()
    
    msg_start_err = "To start a new training, the config must be specified."
    msg_continue_err = "To continue training the config must be 'None'."
    
    if not os.path.isdir(args.logdir): 
        # start new training
        print("starting new training")
        assert args.config, msg_start_err
    
    else: 
        # continue training
        print("continue training")
        assert args.config is None, msg_continue_err
        
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)


def design_pqmf(config: dict):
    # find optimal cutoff_opt
    cutoff, beta = design(torch.tensor(config['pqmf']['length']), 
                          config['melgan']['bands'], 
                          torch.tensor(config['pqmf']['beta']), epochs=10)
    config['pqmf']['cutoff_ratio'] = cutoff
    config['pqmf']['beta'] = beta
    print(f"pqmf design | optimal cutoff-ratio: {cutoff:.4f}")


def train(config, loader, logger, ctx):
    
    device = config["device"]
    ctx.trainer.to(device)
    
    sr = config["data"]["sample_rate"]
    
    t0 = time.time()
    
    for x  in loader:
        
        x = resample(x.to(device), loader.dataset.sr, sr)

        if ctx.total_iters < config["iters_pretrain"]:
            stft_loss = ctx.trainer.pretrain(x)
            D_loss = G_loss = None
        else:
            stft_loss, D_loss, G_loss = ctx.trainer.train(x)
            
        ctx.total_iters += 1 # inc before saving
          
        if ctx.total_iters % args.tracking == 0:
            dt = time.time() - t0
            t0 = time.time()
            
            stft_loss = stft_loss.item()
            D_loss = D_loss.item() if not D_loss is None else D_loss
            G_loss = G_loss.item() if not G_loss is None else G_loss
            
            logger.track_losses(dt, stft_loss, D_loss, G_loss)
            
        if ctx.total_iters % args.saving == 0:
            logger.save_checkpoint()
            logger.save_results()
    
    if ctx.total_iters % args.saving != 0:
        logger.save_checkpoint()
            
                
if __name__ == "__main__":
    
    assert sys.version_info.major == 3
    
    args = parse_args()
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # if DataLoader(drop_last=True)
    #torch.use_deterministic_algorithms(True)
    torch.cuda.empty_cache()
    
    logger = Logger(args.logdir)
    
    if not os.path.isdir(args.logdir) and args.config:
        # starting new training
        
        # config loading
        config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
        
        set_seed(config["seed"])
    
        if config['pqmf']['cutoff_ratio'] is None:
            design_pqmf(config)
            
        # model and data
        ctx = Context(config)
        loader = train_loader(config, args.iters, ctx)
        logger.create(config, ctx)
        
    else: 
        # continue training
        config, ctx = logger.load()
        loader = train_loader(config, args.iters, ctx)
    
    train(config, loader, logger, ctx)
