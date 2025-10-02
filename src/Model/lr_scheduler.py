import math

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = max(warmup_steps, 1)
        self.total_steps = max(total_steps, warmup_steps + 1)
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = 0.5 * self.base_lr * (1.0 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr