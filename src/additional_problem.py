import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision import transforms

from optimizers import RSGD
from optimizers.learning_rate import ConstantLR, DiminishingLR, CosineAnnealingLR, PolynomialLR
from optimizers.batchsize_scheduler import ConstantBS, ExponentialGrowthBS, PolynomialGrowthBS
from manifolds import Manifold, Stiefel, Sphere
from utils import Problem

class AdditionalProblem(Problem):
    def __init__(self, manifold: Manifold, data: np.ndarray) -> None:
        super(AdditionalProblem, self).__init__(manifold)
        self.data = data

    def f(self, point: np.ndarray) -> float:
        s = self.data @ point
        return np.mean(np.abs(s)**(0.5))

    def egrad(self, point: np.ndarray, idx: int) -> np.ndarray:
        s = self.data @ point
        abs_s = np.abs(s)
        coeff = (0.5 /len(s)) * np.sign(s) * np.where(abs_s > 0, abs_s**(-0.5), 0)
        return coeff @ self.data

    def minibatch_grad(self, point: np.ndarray, batch_size: int) -> np.ndarray:
        N = self.data.shape[0]
        idx = np.random.randint(0, N, size=batch_size)
        minibatchX = self.data[idx]
        s = minibatchX @ point
        abs_s = np.abs(s)
        coeff = (0.5 / batch_size) * np.sign(s) * np.where(abs_s > 0, abs_s**(-0.5), 0)
        miniegrad = coeff @ minibatchX
        return self.manifold.projection(point, miniegrad)

    def full_grad(self, point: np.ndarray) -> np.ndarray:
        s = self.data @ point
        abs_s = np.abs(s)
        coeff = (0.5/len(s)) * np.sign(s) * np.where(abs_s > 0, abs_s**(-0.5), 0)
        egrad = coeff @ self.data
        return self.manifold.projection(point, egrad)


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed=seed)
    
    num_all_steps: int = 3000

    lr_min: float = 0.0
    expobs_base: float = 3.0
    
    #lr_type_list = ['ConstantLR', 'DiminishingLR', 'CosAnnealLR', 'PolyLR']
    lr_type_list = ['CosAnnealLR']
    lr_list = [0.01]

    X = np.random.normal(size=(60000, 1024)) # (7200, 1024) or (60000, 1024)
    X /= np.linalg.norm(X, axis=1, keepdims=True) # dataset
    N, n = X.shape
    if N == 7200: 
        init_batch_list = [[3**3, 0], [3**3, 3], [3**3, 6], [3**5, 0], [3**6, 3], [3**8, 0], [30, 6], [800, 3], [7200, 0]]
    elif N == 60000:
        init_batch_list = [[3**5, 0], [3**5, 3], [3**5, 6], [3**7, 0], [3**8, 3], [3**10, 0], [60000, 0], [247, 6], [2223, 3]]

    manifold = Sphere(n=n) # Constraint
    initial = np.random.normal(size=n)
    initial /= np.linalg.norm(initial)  # initial point
    problem = AdditionalProblem(manifold=manifold, data=X) # problem

    print(f'{(N, n)=}')

    for init_batch, num_batch_up in init_batch_list:
        if num_batch_up == 0:
            bs = ConstantBS(init_batch=init_batch, num_all_steps=num_all_steps)
        elif num_batch_up == 3 or num_batch_up == 6:
            bs = ExponentialGrowthBS(init_batch=init_batch, num_all_steps=num_all_steps, num_batch_up=num_batch_up, expo_base=expobs_base)
        else:
            raise ValueError(f'Invalid num_batch_up: {num_batch_up}')
            
        for lr_type in lr_type_list:
            for lr_max in lr_list:
                if lr_type == 'ConstantLR':
                    lr = ConstantLR(base_lr=lr_max)
                elif lr_type == 'CosAnnealLR':
                    lr = CosineAnnealingLR(base_lr=lr_max, lr_min=lr_min, num_all_steps=num_all_steps)
                elif lr_type == 'PolyLR':
                    lr = PolynomialLR(base_lr=lr_max, lr_min=lr_min, num_all_steps=num_all_steps, power=polylr_power)    

                print(f'{(lr_type, lr_max, init_batch, num_batch_up)=}')
                
                pkl_dir: str = os.path.join('rebuttal', 'toyproblem', f'data{N}dim{n}', lr_type)
                if not os.path.isdir(pkl_dir):
                    os.makedirs(pkl_dir)
            
                rsgd=RSGD(bs=bs, lr=lr) 
                history = rsgd.solve(problem, point=initial, num_all_steps=num_all_steps)
                pkl_name: str =  f'{rsgd}-initlr{lr_max}-initbs{init_batch}-num_bsup{num_batch_up}.pkl'
                pd.to_pickle(history, os.path.join(pkl_dir, pkl_name))