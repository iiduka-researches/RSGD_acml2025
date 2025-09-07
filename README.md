# Faster Convergence of Riemannian Stochastic Gradient Descent with Increasing Batch Size
Code for reproducing experiments in our paper.

## Abstruct
We have theoretically analyzed the use of Riemannian stochastic gradient descent (RSGD) and found that using an increasing batch size leads to faster RSGD convergence rate than using a constant batch size not only with a constant learning rate but also with a decaying learning rate, such as cosine annealing decay and polynomial decay. The convergence rate of RSGD improves from $O(T^{-1}+C)$ with a constant batch size to $O(T^{-1})$ with an increasing batch size, where $T$ denotes the total number of iterations and $C$ is a constant. Using principal component analysis and low-rank matrix completion tasks, we investigated, both theoretically and numerically, how increasing batch size affects computational time as measured by stochastic first-order oracle (SFO) complexity. Increasing batch size reduces the SFO complexity of RSGD. Furthermore, our numerical results demonstrated that increasing batch size offers the advantages of both small and large constant batch sizes.

## Acknowledgement
This project is based on the original repository by Sakai Hiroyuki (MIT Licensed).  
The original project is available at: https://github.com/iiduka-researches/202408-adaptive.git
