import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Schedule:
    def __init__(self, optimizer, warmup_epochs, min_lr, factor=0.5, patience=2, threshold=1e-1, verbose=False):     
        # optimizer: the optimizer being used
        # warmup_epochs: number of warmup epochs
        # min_lr: the minimum learning rate after decay
        # factor: the multiplicative factor to reduce LR
        # patience: number of epochs to wait before reducing LR
        # threshold: minimum change to qualify as an improvement
        # verbose: if True, prints learning rate adjustments
        
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.verbose = verbose
        self.current_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.reduce_on_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, threshold=threshold, min_lr=min_lr, verbose=verbose)

    def step(self, metric=None):
        # Warmup phase
        if self.current_epoch < self.warmup_epochs:
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * (self.current_epoch + 1) / self.warmup_epochs
            if self.verbose:
                print(f"Warmup: Setting learning rate to {[group['lr'] for group in self.optimizer.param_groups]}")
        else:
            # After warmup, use ReduceLROnPlateau to adjust learning rate
            if metric is not None:
                self.reduce_on_plateau.step(metric)

        self.current_epoch += 1
        