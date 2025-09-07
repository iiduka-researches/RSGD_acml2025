import numpy as np
from abc import ABC, abstractmethod

class LearningRate(ABC):
    def __init__(self, base_lr: float) -> None:
        if not 0.0 <= base_lr:
            raise ValueError(f'Invalid base learning rate: {base_lr}')
        self.base_lr = base_lr

    def __call__(self, k: int):
        return self.get_lr(k=k)
    
    @abstractmethod
    def get_name(self) -> str:
        ...
    
    @abstractmethod
    def get_lr(self, k: int) -> float:
        ...


class ConstantLR(LearningRate):
    def __init__(self, base_lr: float) -> None:
        super(ConstantLR, self).__init__(base_lr)
    
    def __repr__(self) -> str:
        return str(self.base_lr)
    
    def get_name(self) -> str:
        return 'ConstantLR'
    
    def get_lr(self, k: int) -> float:
        return self.base_lr


class DiminishingLR(LearningRate):
    def __init__(self, base_lr: float) -> None:
        super(DiminishingLR, self).__init__(base_lr)
    
    def __repr__(self) -> str:
        return str(self.base_lr) + ' / sqrt(k + 1)'
    
    def get_name(self) -> str:
        return 'DiminishingLR'

    def get_lr(self, k: int) -> float:
        return self.base_lr / np.sqrt(k + 1)

class CosineAnnealingLR(LearningRate):
    def __init__(self, base_lr: float, lr_min: float, num_all_steps: int) -> None:
        super(CosineAnnealingLR, self).__init__(base_lr)
        if not 0.0 <= lr_min:
            raise ValueError(f'Invalid minimal learning rate: {lr_min}')
        if not 0 < num_all_steps:
            raise ValueError(f'Invalid steps per epoch: {num_all_steps}')
        self.lr_min = lr_min
        self.num_all_steps = num_all_steps
        
    def __repr__(self) -> str:
        return str(self.lr_min) + ' + (' + str(self.base_lr) + ' - ' + str(self.lr_min) + ')' + '* ( 1 + cos(k*pi/N)) / 2'
    
    def get_name(self) -> str:
        return 'CosAnnealLR'
    
    def get_lr(self, k: int) -> float:
        return self.lr_min + (self.base_lr - self.lr_min) * (1 + np.cos(k * np.pi / self.num_all_steps)) / 2
    

class PolynomialLR(LearningRate):
    def __init__(self, base_lr: float, lr_min: float, num_all_steps: int, power:float) -> None:
        super(PolynomialLR, self).__init__(base_lr)
        if not 0.0 <= lr_min:
            raise ValueError(f'Invalid minimal learning rate: {lr_min}')
        if not 0 < num_all_steps:
            raise ValueError(f'Invalid steps per epoch: {num_all_steps}')        
        if not 0.0 < power:
            raise ValueError(f'Invalid number of power: {power}')        
        self.lr_min = lr_min
        self.power = power
        self.num_all_steps = num_all_steps
    
    def __repr__(self) -> str:
        return str(self.lr_min) + ' + (' + str(self.base_lr) + ' - ' + str(self.lr_min) + ') * (1 - t / T)^p'
    
    def get_name(self) -> str:
        return 'PolyLR'
    
    def get_lr(self, k: int) -> float:
        return self.lr_min + (self.base_lr - self.lr_min) * ((1 - k / self.num_all_steps)**(self.power))


class ExponentialGrowthLR(LearningRate):
    def __init__(self, base_lr: float, num_all_steps: int, num_lr_up: int, expo_base:float) -> None:
        super(ExponentialGrowthLR, self).__init__(base_lr)
        if not 1 <= num_lr_up:
            raise ValueError(f'Invalid number of times learning rate is increased: {num_lr_up}')
        if not 1 < expo_base:
            raise ValueError(f'(exponential) Base must be greater than 1: {expo_base}')
        self.num_all_steps = num_all_steps
        self.num_lr_up = num_lr_up
        self.expo_base = expo_base
        self.steps_one_lr = int(self.num_all_steps / self.num_lr_up)
        
    def __repr__(self) -> str:
        return str(self.base_lr) + ' * (' + str(self.expo_base) + ' ** m)'
    
    def get_name(self) -> str:
        return 'ExpoGrowthLR'
    
    def get_lr(self, k: int) -> float:
        m = int(k / self.steps_one_lr)
        if m == self.num_lr_up:
            m = self.num_lr_up -1
        
        return self.base_lr * (self.expo_base ** m)


class PolynomialGrowthLR(LearningRate):
    def __init__(self, base_lr: float, num_all_steps: int, num_lr_up: int, slope: float, power: float) -> None:
        super(PolynomialGrowthLR, self).__init__(base_lr)
        if not 1 <= num_lr_up:
            raise ValueError(f'Invalid number of times learning rate is increased: {num_lr_up}')
        if not slope > 0:
            raise ValueError(f'Slope must be greater than 0: {slope}')
        if not power > 1:
            raise ValueError(f'Power of poly bs must be greater than 1: {power}')
        self.num_all_steps = num_all_steps
        self.num_lr_up = num_lr_up
        self.slope = slope
        self.power = power
        self.steps_one_lr = int(self.num_all_steps / self.num_lr_up)
        
    def __repr__(self) -> str:
        return '(' + str(self.slope) + ' * m + ' + str(self.base_lr) + ')**' + str(self.power)
    
    def get_name(self) -> str:
        return 'PolyGrowthLR'
    
    def get_lr(self, k: int) -> float:
        m = int(k / self.steps_one_lr)
        if m == self.num_lr_up:
            m = self.num_lr_up - 1
        
        return (self.slope * m + self.base_lr)**self.power
    

# For only the case num_lr_up = 3, 8
class ExponentialWarmUpLR(LearningRate):
    def __init__(self, base_lr: float, num_all_steps: int, warm_up_rate: float, num_lr_up: int, lr_max:float, decay_part_name:str) -> None:
        super(ExponentialWarmUpLR, self).__init__(base_lr)
        if not 1 <= num_lr_up:
            raise ValueError(f'Invalid number of times learning rate is increased: {num_lr_up}')
        self.num_all_steps = num_all_steps
        self.num_warm_up_steps = int(num_all_steps * warm_up_rate)
        self.num_lr_up = num_lr_up
        self.lr_max = lr_max
        self.steps_one_lr = int(self.num_all_steps / self.num_lr_up)
        
        flag = 0
        for name in ["ConstantLR", "DiminishingLR", "CosAnnealLR", "PolyLR"]:
            if decay_part_name == name:
                flag +=1
        if not flag == 1:
            raise NameError(f'Invalid learning rate for decaying part: {decay_part_name}')
        self.decay_part_name = decay_part_name
        
        if self.num_lr_up == 3:
            self.lr = ExponentialGrowthLR(base_lr=self.base_lr, num_all_steps=self.num_warm_up_steps, num_lr_up=self.num_lr_up, expo_base=1.193)
        elif self.num_lr_up == 8:
            self.lr = ExponentialGrowthLR(base_lr=self.base_lr, num_all_steps=self.num_warm_up_steps, num_lr_up=self.num_lr_up, expo_base=1.189)
            
    def get_name(self) -> str:
        return f"ExpoWarmUpLRand{self.decay_part_name}"
    
    def get_lr(self, k:int) -> float:
        if k == self.num_warm_up_steps - 2:
            self.x_tw = self.lr.get_lr(k)
        if k == self.num_warm_up_steps - 1:
            if self.decay_part_name == "ConstantLR":
                self.lr = ConstantLR(base_lr=self.x_tw)
            elif self.lr == "DiminishingLR":
                self.lr = DiminishingLR(base_lr=self.x_tw)
            elif self.decay_part_name == "CosAnnealLR":
                self.lr = CosineAnnealingLR(base_lr=self.x_tw, lr_min=0.0, num_all_steps=self.num_all_steps-self.num_warm_up_steps)
            elif self.decay_part_name == "PolyLR":
                self.lr = PolynomialLR(base_lr=self.x_tw, lr_min=0.0, num_all_steps=self.num_all_steps-self.num_warm_up_steps, power=2.0)
        
        return self.lr.get_lr(k)
