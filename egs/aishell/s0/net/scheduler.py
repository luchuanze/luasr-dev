
import torch
from torch.optim.lr_scheduler import _LRScheduler


class LRScheduler(_LRScheduler):
    """
    The warmup LR scheduler
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int = 20000,
                 last_step: int = -1):
        self.warmup_steps = warmup_steps

        super().__init__(optimizer, last_step)

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr
            * self.warmup_steps ** 0.5
            * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
            for lr in self.base_lrs
        ]

    def set_step(self, step: int):
        self.last_epoch = step