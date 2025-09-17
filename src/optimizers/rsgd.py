import numpy as np
from time import time
from tqdm import tqdm

from utils import Problem
from .optimizer import Optimizer
from .learning_rate import LearningRate
from .batchsize_scheduler import BatchSizeScheduler


class RSGD(Optimizer):
    '''
    Implements Riemannian Stochastic Grafient Descent.

    Attribute
    ---------
    batch_size (int):
        batch size.
    lr (LearningRate):
        learning rate.
    '''
    #completed
    def __init__(
            self,
            bs: BatchSizeScheduler,
            lr: LearningRate,
        ) -> None:
        super(RSGD, self).__init__(bs=bs, lr=lr)
    
    def __repr__(self) -> str:
        return 'RSGD-' + self.bs.get_name() + self.lr.get_name()
    
    #ここで問題解く
    #停止条件は反復回数が指定した値max_iterに達すること
    def solve(self, problem: Problem, point: np.ndarray, num_all_steps: int):
        self.init_history()
        self.update_history(
            loss=problem.f(point),
            test_loss=problem.f_test(point),
            grad_norm=np.linalg.norm(problem.full_grad(point)),
            test_grad_norm=np.linalg.norm(problem.full_grad_test(point)),
            elapsed_time=None
        )
        '''use this unless ..._withtest.py
        self.update_history(
            loss=problem.f(point),
            grad_norm=np.linalg.norm(problem.full_grad(point)),
            elapsed_time=None
        )
        '''

        #ここでアルゴリズムを回している。
        for k in tqdm(range(max_itr)):
            start_time = time()
            grad = problem.minibatch_grad(point, int(self.bs(k))) # calculate minibatch grad
            d_p = -self.lr(k) * grad # __call__: self.lr(k)
            point = problem.manifold.retraction(point, d_p)
            end_time = time()

            self.update_history(
                loss=problem.f(point),
                test_loss=problem.f_test(point),
                grad_norm=np.linalg.norm(problem.full_grad(point)),
                test_grad_norm=np.linalg.norm(problem.full_grad_test(point)),
                elapsed_time=end_time - start_time
            )
            '''use this unless ..._withtest.py
            self.update_history(
                loss=problem.f(point),
                grad_norm=np.linalg.norm(problem.full_grad(point)),
                elapsed_time=end_time - start_time
            )
            ''' 
        
        #self.history: optimizer class内にある。アルゴリズムの更新の情報を持った辞書
        return self.history
