import numpy as np
from abc import ABC, abstractmethod

class BatchSizeScheduler(ABC):
    def __init__(self, init_batch: int, num_all_steps: int) -> None:
        if not 1 <= init_batch:
            raise ValueError(f'Invalid initial batch size: {init_batch}')
        if not 1 <= num_all_steps:
            raise ValueError(f'Invalid width of batch size scheduling: {num_all_steps}')
        self.init_batch = init_batch
        self.num_all_steps = num_all_steps
        
    def __call__(self, k:int) -> int:
        return self.get_bs(k=k)
    
    @abstractmethod
    def get_name(self) -> str:
        ...
        
    @abstractmethod
    def get_bs(self, k:int) -> int:
        ...
        
class ConstantBS(BatchSizeScheduler):
    def __init__(self, init_batch: int, num_all_steps: int) -> None:
        super(ConstantBS, self).__init__(init_batch, num_all_steps)
    
    def __call__(self, k: int) -> int:
        return self.init_batch
    
    def __repr__(self) -> str:
        return str(self.init_batch)
    
    def get_name(self) -> str:
        return 'ConstantBS'
    
    def get_bs(self, k: int) -> int:
        return self.init_batch

class ExponentialGrowthBS(BatchSizeScheduler):
    def __init__(self, init_batch: int, num_all_steps: int, num_batch_up: int, expo_base: float) -> None:
        super(ExponentialGrowthBS, self).__init__(init_batch, num_all_steps)
        if not 1 <= num_batch_up:
            raise ValueError(f'Invalid number of times batch size is increased: {num_batch_up}')
        if not 1 < expo_base:
            raise ValueError(f'(exponential) Base must be greater than 1: {expo_base}')
        self.num_batch_up = num_batch_up
        self.expo_base = expo_base
        self.steps_one_batch = int(self.num_all_steps / self.num_batch_up)
        
    def __repr__(self) -> str:
        return str(self.init_batch) + ' * (' + str(self.expo_base) + ' ** m)'
    
    def get_name(self) -> str:
        return 'ExpoGrowthBS'
    
    def get_bs(self, k:int) -> int:
        m = int(k / self.steps_one_batch)
        if m == self.num_batch_up:
            m = self.num_batch_up - 1
                    
        return self.init_batch * (self.expo_base ** m)
        
class PolynomialGrowthBS(BatchSizeScheduler):
    def __init__(self, init_batch: int, num_all_steps: int, num_batch_up: int, slope: float, power: float) -> None:
        super(PolynomialGrowthBS, self).__init__(init_batch, num_all_steps)
        if not 1 <= num_batch_up:
            raise ValueError(f'Invalid number of times batch size is increased: {num_batch_up}')
        if not slope > 0:
            raise ValueError(f'Slope must be greater than 0: {slope}')
        if not power > 1:
            raise ValueError(f'Power of poly bs must be greater than 1: {power}')
        self.num_batch_up = num_batch_up
        self.slope = slope
        self.power = power
        self.steps_one_batch = int(self.num_all_steps / self.num_batch_up)        
    
    def __repr__(self) -> str:
        return '(' + str(self.slope) + ' * m + ' + str(self.init_batch) + ')**' + str(self.power)
    
    def get_name(self) -> str:
        return 'PolyGrowthBS'
    
    def get_bs(self, k:int) -> int:
        m = int(k / self.steps_one_batch)
        if m == self.num_batch_up:
            m = self.num_batch_up - 1
        
        return (self.slope * m + self.init_batch)**self.power
    