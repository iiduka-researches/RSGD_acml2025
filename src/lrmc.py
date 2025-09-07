import argparse
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt

from optimizers import RSGD
from optimizers.learning_rate import ConstantLR, DiminishingLR, CosineAnnealingLR, PolynomialLR
from optimizers.batchsize_scheduler import ConstantBS, ExponentialGrowthBS, PolynomialGrowthBS
from manifolds import Manifold, Grassmann
from utils import Problem

class LowRankMatrixCompletion(Problem):
    def __init__(
        self,
        manifold: Manifold,
        data: np.ndarray
    ) -> None:
        super(LowRankMatrixCompletion, self).__init__(manifold)
        self.data = data
        
    def f(self, point: np.ndarray) -> float:
        X = self.data
        N: int = X.shape[0]
    
        _sum: float = 0.
        for idx in range(N):
            x = np.zeros(point.shape[0])
            nonzero_indices: list[int] = X.indices[X.indptr[idx]: X.indptr[idx + 1]]
            nonzero_values = X.data[X.indptr[idx]: X.indptr[idx + 1]]
            x[nonzero_indices] = nonzero_values

            U_omega = point[nonzero_indices]
            a = np.linalg.lstsq(U_omega, nonzero_values, rcond=None)[0]

            indicator = np.zeros(point.shape[0])
            indicator[nonzero_indices] = 1.

            _sum += np.linalg.norm(indicator * (point @ a) - x) ** 2
        return _sum / N

    def egrad(self, point: np.ndarray, idx: int) -> np.ndarray:
        X = self.data
        x = np.zeros(point.shape[0])
        nonzero_indices: list[int] = X.indices[X.indptr[idx]: X.indptr[idx + 1]]
        nonzero_values = X.data[X.indptr[idx]: X.indptr[idx + 1]]
        x[nonzero_indices] = nonzero_values

        U_omega = point[nonzero_indices]
        a = np.linalg.lstsq(U_omega, nonzero_values, rcond=None)[0]

        indicator = np.zeros(point.shape[0])
        indicator[nonzero_indices] = 1.

        return np.expand_dims((indicator * (point @ a) - x), axis=1) @ np.expand_dims(a, axis=1).T

    def minibatch_grad(
        self,
        point: np.ndarray,
        batch_size: int
    ) -> np.ndarray:
        X = self.data
        N: int = X.shape[0]
        egrad = np.zeros_like(point)
        for _ in range(batch_size):
            idx = np.random.randint(0, N)
            egrad += self.egrad(point, idx)
        return self.manifold.projection(point, egrad) / batch_size / 2

    def full_grad(
        self,
        point: np.ndarray,
    ) -> np.ndarray:
        X = self.data
        N: int = X.shape[0]
        egrad = np.zeros_like(point)
        for idx in range(N):
            egrad += self.egrad(point, idx)
        return self.manifold.projection(point, egrad) / N / 2
    
if __name__ == '__main__':
    seed = 0
    np.random.seed(seed=seed)

    num_batch_up: int = 3
    p: int = 10 # default: int = 10
    
    lr_min: float = 0.0
    polybs_power: float = 2.0
    polylr_power: float = 2.0
    #init_batch: int = 2**8 # default:2**8
    
    data_list = ['ml-1m', 'jester']
    batch_type_list = ['ConstantBS', 'ExpoGrowthBS', 'PolyGrowthBS']
    lr_type_list = ['ConstantLR', 'DiminishingLR', 'CosAnnealLR', 'PolyLR']
    lr_list = [0.5, 0.1, 0.05, 0.01, 0.005]
    for data_set in data_list:
        if data_set == 'ml-1m':
            df = pd.read_csv('data/ml/ml-1m/ratings.csv')
            data = csr_matrix((df.Rating, (df.MovieID - 1, df.UserID - 1))) / 5
        elif data_set == 'jester':
            df = pd.read_csv('data/jester/jester_1.csv')
            data = csr_matrix((df.data, (df.col, df.row)))
        
        N: int = data.shape[0]
        n: int = data.shape[1]
        print(f'{(N, n, p)=}')

        initial = np.linalg.qr(np.random.rand(n, p))[0]
        manifold = Grassmann(p, n)
        problem = LowRankMatrixCompletion(manifold=manifold, data=data)

        for batch_type in batch_type_list:
            if batch_type == 'ConstantBS':
                bs = ConstantBS(init_batch=2**8, num_all_steps=N)
            elif batch_type == 'ExpoGrowthBS':
                bs = ExponentialGrowthBS(init_batch=3**4, num_all_steps=N, num_batch_up=num_batch_up, expo_base=3.0)
            elif batch_type == 'PolyGrowthBS':
                bs = PolynomialGrowthBS(init_batch=3**4, num_all_steps=N, num_batch_up=num_batch_up, slope=2.0, power=polybs_power)
                
            for lr_type in lr_type_list:
                for lr_max in lr_list:
                    print(f'{(data_set, batch_type, lr_type, lr_max)=}')

                    pkl_dir: str = os.path.join('results', os.path.join(data_set, os.path.join(batch_type, lr_type)))
                    if not os.path.isdir(pkl_dir):
                        os.makedirs(pkl_dir)

                    if lr_type == 'ConstantLR':
                        lr = ConstantLR(base_lr=lr_max)
                    elif lr_type == 'DiminishingLR':
                        lr = DiminishingLR(base_lr=lr_max)
                    elif lr_type == 'CosAnnealLR':
                        lr = CosineAnnealingLR(base_lr=lr_max, lr_min=0.0, num_all_steps=N)
                    elif lr_type == 'PolyLR':
                        lr = PolynomialLR(base_lr=lr_max, lr_min=0.0, num_all_steps=N, power=polylr_power)
                
                        
                    rsgd=RSGD(bs=bs, lr=lr)
                    history = rsgd.solve(problem, point=initial, N=N)
                    pkl_name: str =  f'{rsgd}-initlr{lr_max}.pkl'
                    pd.to_pickle(history, os.path.join(pkl_dir, pkl_name))