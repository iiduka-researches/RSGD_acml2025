# This code requires numpy version < 2.

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms

from optimizers import RSGD
from optimizers.learning_rate import ExponentialWarmUpLR
from optimizers.batchsize_scheduler import ConstantBS, ExponentialGrowthBS, PolynomialGrowthBS
from manifolds import Manifold, Stiefel
from utils import Problem

class PrincipalComponentAnalysis(Problem):
    def __init__(
        self,
        manifold: Manifold,
        data: np.ndarray
    ) -> None:
        super(PrincipalComponentAnalysis, self).__init__(manifold)
        self.data = data
        
    def f(self, point: np.ndarray) -> float:
        X = self.data
        N: int = X.shape[0]
        return np.linalg.norm(X.T - point @ point.T @ X.T) ** 2 / N

    def egrad(self, point: np.ndarray, idx: int) -> np.ndarray:
        X = self.data
        x: np.ndarray = X[idx]
        _x = np.expand_dims(x, axis=1)
        return -2 * (_x @ _x.T @ point)

    def minibatch_grad(
        self,
        point: np.ndarray,
        batch_size: int
    ) -> np.ndarray:
        N: int = self.data.shape[0]
        samples = [np.random.randint(0, N) for _ in range(batch_size)]
        X = self.data[samples]
        return -2 * self.manifold.projection(point, (X.T @ X @ point) / batch_size)

    def full_grad(
        self,
        point: np.ndarray,
    ) -> np.ndarray:
        X = self.data
        N: int = X.shape[0]
        return -2 * self.manifold.projection(point, (X.T @ X @ point) / N)

if __name__ == '__main__':
    seed = 0
    np.random.seed(seed=seed)
    
    num_batch_up: int = 5
    num_all_steps: int = 3000

    lr_min: float = 0.0
    expobs_base: float = 3.0
    polybs_power: float = 2.0
    polylr_power: float = 2.0
    init_batch: int = 3**4 # default: 2**10        
    
    data_list = ['COIL100', 'MNIST']
    batch_type_list = ['ExpoGrowthBS', 'PolyGrowthBS']
    lr_decay_part_list = ['ConstantLR', 'DiminishingLR', 'CosAnnealLR', 'PolyLR']
    num_lr_up_list = [3, 8]
    lr_max_list = [0.5, 0.05, 0.005]
    base_lr_list = [0.294, 0.0294, 0.00294, 0.125, 0.0125, 0.00125]
    for dataset in data_list:
        if dataset == 'MNIST':
            data = MNIST(root="data", download=True, train=True, transform=transforms.ToTensor())
            data = data.data.view(-1, 28 * 28) / 255
            data = data.numpy().copy()
            N: int = data.shape[0]
            n: int = data.shape[1]
            p: int = 10
        elif dataset == 'COIL100':
            data = pd.read_csv('data/coil100/coil100_grayscale.csv', header=None).values / 255
            N: int = data.shape[0]
            n: int = data.shape[1]
            p: int = 100
        print(f'{(N, n, p)=}')
        
        manifold = Stiefel(p, n)
        initial = np.linalg.qr(np.random.rand(n, p))[0]
        problem = PrincipalComponentAnalysis(manifold=manifold, data=data)

        for batch_type in batch_type_list:
            if batch_type == 'ConstantBS':
                bs = ConstantBS(init_batch=2**10, num_all_steps=num_all_steps)
            elif batch_type == 'ExpoGrowthBS':
                bs = ExponentialGrowthBS(init_batch=init_batch, num_all_steps=num_all_steps, num_batch_up=num_batch_up, expo_base=expobs_base)
            elif batch_type == 'PolyGrowthBS':
                bs = PolynomialGrowthBS(init_batch=init_batch, num_all_steps=num_all_steps, num_batch_up=num_batch_up, slope=2.0, power=polybs_power)
            
            for num_lr_up in num_lr_up_list:
                for lr_decay_part in lr_decay_part_list:
                    for base_lr in base_lr_list:
                        for lr_max in lr_max_list:
                            print(f'{(dataset, batch_type, lr_decay_part, num_lr_up, base_lr, lr_max)=}')
                            pkl_dir: str = os.path.join('results', dataset, batch_type, "WarmUpLR", "lr_up:" + str(num_lr_up), lr_decay_part)
                            if not os.path.isdir(pkl_dir):
                                os.makedirs(pkl_dir)
                            
                            if num_lr_up == 3:
                                if (base_lr == 0.294 and lr_max == 0.5) or (base_lr == 0.0294 and lr_max == 0.05) or (base_lr == 0.00294 and lr_max == 0.005):
                                    lr = ExponentialWarmUpLR(base_lr=base_lr, num_all_steps=num_all_steps, warm_up_rate=0.2, num_lr_up=num_lr_up, lr_max=lr_max, decay_part_name=lr_decay_part)
                                else:
                                    continue
                            elif num_lr_up == 8:
                                if (base_lr == 0.125 and lr_max == 0.5) or (base_lr == 0.0125 and lr_max == 0.05) or (base_lr == 0.00125 and lr_max == 0.005):
                                    lr = ExponentialWarmUpLR(base_lr=base_lr, num_all_steps=num_all_steps, warm_up_rate=0.53, num_lr_up=num_lr_up, lr_max=lr_max, decay_part_name=lr_decay_part)
                                else:
                                    continue
            
                            rsgd=RSGD(bs=bs, lr=lr)
                            history = rsgd.solve(problem, point=initial, N=num_all_steps)
                            pkl_name: str =  f'{rsgd}-lrmax{lr_max}.pkl'
                            pd.to_pickle(history, os.path.join(pkl_dir, pkl_name))