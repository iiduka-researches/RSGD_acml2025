from abc import ABC, abstractmethod
from .learning_rate import LearningRate
from .batchsize_scheduler import BatchSizeScheduler

class Optimizer(ABC):
    '''
    Base class for all optimizers.

    Attribute
    ---------
    bs (BatchSizeScheduler):
        batch size.
    lr (LearningRate):
        learning rate.
    '''
    def __init__(self, bs: BatchSizeScheduler, lr: LearningRate) -> None:
        self.bs = bs
        self.lr = lr
        self.history = dict()
    
    def init_history(self) -> None:
        self.history = dict(
            optimizer=repr(self),
            batch_size=self.bs.init_batch,
            batch_type=self.bs.get_name(),
            lr=self.lr,
            loss=[],
            grad_norm=[],
            elapsed_time=[]
        )
    
    def update_history(self, loss: float=None, grad_norm: float=None, elapsed_time: float=None) -> None:
        if loss:
            self.history['loss'].append(loss)
        if grad_norm:
            self.history['grad_norm'].append(grad_norm)
        if elapsed_time:
            self.history['elapsed_time'].append(elapsed_time)

    @abstractmethod
    def solve():
        ...
