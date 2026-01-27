import math
from torch.optim.lr_scheduler import _LRScheduler

class ChainCyclicLR(_LRScheduler):
    def __init__(self, optimizer, lr_min, lr_max, warmup_steps, T0, scale_factor=1.0, last_epoch=-1):
        """
        optimizer: torch.optim.Optimizer
        lr_min: минимальный lr
        lr_max: максимальный lr первой волны
        warmup_steps: количество шагов линейного разгона
        T0: количество шагов на полный цикл (без учёта следующей волны)
        scale_factor: коэффициент увеличения lr_max для следующей волны
        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.T0 = T0
        self.scale_factor = scale_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # какой шаг в текущем цикле
        step_in_cycle = self.last_epoch % self.T0

        if step_in_cycle < self.warmup_steps:
            # линейный рост
            lr = self.lr_min + (self.lr_max - self.lr_min) * step_in_cycle / self.warmup_steps
        else:
            # спад: используем косинус для плавности
            progress = (step_in_cycle - self.warmup_steps) / (self.T0 - self.warmup_steps)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))
        
        # корректируем для новых волн
        # каждая новая волна увеличивает lr_max на scale_factor^(номер цикла)
        cycle_number = self.last_epoch // self.T0
        lr = min(lr * (self.scale_factor ** cycle_number), self.lr_max * (self.scale_factor ** cycle_number))
        
        return [lr for _ in self.optimizer.param_groups]
