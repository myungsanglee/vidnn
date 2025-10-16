import math
from bisect import bisect_left
import numpy as np
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """
    https://github.com/gaussian37/pytorch_deep_learning_models/blob/master/cosine_annealing_with_warmup/cosine_annealing_with_warmup.py
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [
                base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class YoloLR(_LRScheduler):
    def __init__(self, optimizer, burn_in, steps, scales, last_epoch=-1):
        self.burn_in = burn_in
        self.steps = steps
        self.scales = scales
        self.scale = 1.0
        super(YoloLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.burn_in:
            return [base_lr * pow(self.last_epoch / self.burn_in, 4) for base_lr in self.base_lrs]
        else:
            if self.last_epoch < self.steps[0]:
                return self.base_lrs
            else:
                if self.last_epoch in self.steps:
                    self.scale *= self.scales[bisect_left(self.steps, self.last_epoch)]
                return [base_lr * self.scale for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class VidnnScheduler(_LRScheduler):
    """
    A custom learning rate scheduler that combines a step-based warmup phase with an epoch-based decay phase (linear or cosine).

    This scheduler replicates the behavior used in the Ultralytics training pipeline.
    It must be called at every `step`.

    Args:
        optimizer (Optimizer): The optimizer to wrap.
        total_epochs (int): The total number of training epochs.
        steps_per_epoch (int): The number of steps (batches) in one epoch.
        warmup_steps (int): The total number of steps for the warmup phase.
        lrf (float, optional): The final learning rate factor (final_lr = initial_lr * lrf). Defaults to 0.01.
        cos_lr (bool, optional): If True, uses cosine annealing for the decay phase. Otherwise, uses linear decay. Defaults to False.
        warmup_momentum (float, optional): The initial momentum for the warmup phase. Defaults to 0.8.
        warmup_bias_lr (float, optional): The initial learning rate for the bias parameter group during warmup. Defaults to 0.1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        steps_per_epoch: int,
        warmup_steps: int,
        lrf: float = 0.01,
        cos_lr: bool = False,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.1,
    ):

        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = warmup_steps
        self.lrf = lrf
        self.cos_lr = cos_lr
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr

        # Main decay function (epoch-based)
        if self.cos_lr:
            # Cosine decay lambda
            self.lr_lambda = self._cosine_lr_scheduler
        else:
            # Linear decay lambda
            self.lr_lambda = self._linear_lr_scheduler

        super().__init__(optimizer)

    def _linear_lr_scheduler(self, epoch):
        """Calculates linear learning rate decay."""
        return max(1 - epoch / self.total_epochs, 0) * (1.0 - self.lrf) + self.lrf

    def _cosine_lr_scheduler(self, epoch):
        """Calculates cosine learning rate decay."""
        # Replicates _one_cycle(y1=1.0, y2=self.lrf, steps=self.total_epochs)
        y1 = 1.0
        y2 = self.lrf
        steps = self.total_epochs
        return max((1 - math.cos(epoch * math.pi / steps)) / 2, 0) * (y2 - y1) + y1

    def get_lr(self):
        """
        Calculates the learning rate for the current step.
        This method is called by `_LRScheduler`'s `step()` method.
        Note: `self.last_epoch` is used here as a step counter.
        """
        current_step = self.last_epoch

        if current_step <= self.warmup_steps:
            # --- Warmup Phase (step-based) ---
            interp_values = [0, self.warmup_steps]

            new_lrs = []
            for i, (base_lr, param_group) in enumerate(zip(self.base_lrs, self.optimizer.param_groups)):
                # The first param group is assumed to be the bias group
                start_lr = self.warmup_bias_lr if i == 0 else 0.0

                # The target LR is the one calculated by the main epoch-based scheduler for epoch 0
                current_epoch = current_step // self.steps_per_epoch
                target_lr_epoch0 = base_lr * self.lr_lambda(current_epoch)

                lr = np.interp(current_step, interp_values, [start_lr, target_lr_epoch0])
                new_lrs.append(lr)

                if "momentum" in param_group:
                    base_momentum = self.optimizer.defaults["momentum"]
                    param_group["momentum"] = np.interp(current_step, interp_values, [self.warmup_momentum, base_momentum])

            return new_lrs
        else:
            # --- Decay Phase (epoch-based) ---
            # Adjust step count to be relative to the start of the decay phase
            # decay_step = current_step - self.warmup_steps
            # current_epoch = decay_step // self.steps_per_epoch
            current_epoch = current_step // self.steps_per_epoch
            multiplier = self.lr_lambda(current_epoch)
            return [base_lr * multiplier for base_lr in self.base_lrs]

    # def step(self, epoch=None):
    #     """
    #     Performs a scheduler step. Also handles momentum updates during warmup.
    #     Note: The `epoch` argument is ignored; this scheduler is purely step-driven.
    #     """
    #     # current_step = self.last_epoch + 1

    #     # Update momentum during warmup
    #     # if current_step < self.warmup_steps:
    #     #     interp_values = [0, self.warmup_steps]
    #     #     for i, param_group in enumerate(self.optimizer.param_groups):
    #     #         if "momentum" in param_group:
    #     #             base_momentum = self.optimizer.defaults.get("momentum", 0.9)
    #     #             param_group["momentum"] = np.interp(current_step, interp_values, [self.warmup_momentum, base_momentum])

    #     # if current_step < self.warmup_steps:
    #     #     if "momentum" in self.optimizer.defaults:
    #     #         interp_values = [0, self.warmup_steps]
    #     #         base_momentum = self.optimizer.defaults["momentum"]
    #     #         for param_group in self.optimizer.param_groups:
    #     #             param_group["momentum"] = np.interp(current_step, interp_values, [self.warmup_momentum, base_momentum])

    #     super().step()
