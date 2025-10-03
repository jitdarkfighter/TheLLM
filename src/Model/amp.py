"""
Automatic Mixed Precision (amp) for PyTorch
Deep Neural Network training has traditionally relied on IEEE single-precision format, however with mixed precision, 
you can train with half precision while maintaining the network accuracy achieved with single precision. 
This technique of using both single- and half-precision representations is referred to as mixed precision technique. 

It speeds up memory limited operations and reduces memory requirement, allowing for larger models or larger batch sizes.
"""

import torch

class ampGrad:
    def __init__(self, optimizer, accumulation_steps: int = 1):
        self.optimizer = optimizer
        self.accumulation_steps = max(1, accumulation_steps)
        self.amp = torch.cuda.is_available()  # Enable AMP if CUDA is available
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self._step = 0

    def backward(self, loss: torch.Tensor):
        loss = loss / self.accumulation_steps
        self.scaler.scale(loss).backward()
        self._step += 1

    def should_step(self):
        return self._step % self.accumulation_steps == 0

    def step(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def zero_grad(self):
        self.optimizer.zero_grad()

    
