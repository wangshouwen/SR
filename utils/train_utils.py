import os
import shutil

import numpy as np
import torch


def adjust_learning_rate(args, epoch, optimizer):
    """ Sets learning rate if it is needed """
    lr_list = []
    decay = 0.1 if sum(epoch == np.array(args.epoch_step)) > 0 else 1.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay
        lr_list.append(param_group['lr'])

    return np.unique(lr_list)


def save_checkpoint(checkpoint, model_dir, is_best=False, epoch=1000):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    filename = f'checkpoint{epoch}.pth'
    res_path = os.path.join(model_dir, filename)
    print('Save checkpoint to {}'.format(res_path))
    torch.save(checkpoint, res_path)
    if is_best:
        filename_best = 'checkpoint_best.pth'
        res_path_best = os.path.join(model_dir, filename_best)
        shutil.copyfile(res_path, res_path_best)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)