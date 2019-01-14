import os
import random
import numpy as np
from scipy import optimize
from multiprocessing import Pool
import LengthToGamma as lg


class OptimalGeometry:

    def __init__(self, notch_count=5, lower_bound=4, upper_bound=40,
                 init_len=1, max_iter=100, model=optimize.dual_annealing):

        print("__init__ Optimal Geometry")
        self.notch_count = notch_count
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.init_len = init_len
        self.max_iter = max_iter
        self.model = model

        self.init_len_list = self.initial_random_notch_geometry()
        self.bounds = self.notch_bounds()
        self.args = self.gen_args()

    def notch_bounds(self):
        return list(zip([self.lower_bound]*self.notch_count,[self.upper_bound]*self.notch_count))

    def gamma(self, lengths):
        return -lg.LengthToGamma(lengths)

    def initial_random_notch_geometry(self):
        values = np.linspace(self.lower_bound,self.upper_bound, 37).astype(int)
        x0_list=[]
        for starting_point in range(self.init_len):
            x0=[]
            for i in range(self.notch_count):
               x0.append(random.choice(values))
            x0_list.append(x0)
        return x0_list

    def minimize(self,args):
        f, model, bounds, max_iter, x0, mutation_val = args

        if self.model is optimize.differential_evolution:
            res = model(f, bounds=bounds, maxiter=max_iter,mutation=(0, mutation_val))
        elif self.model is optimize.dual_annealing:
            res = model(f, bounds=bounds, maxiter=max_iter, x0=x0)
        return -res.fun, list(res.x.astype(int))

    def gen_args(self):
        return [(self.gamma, self.model, self.bounds, self.max_iter, self.init_len_list[i], 2 * np.random.rand()) for i in range(self.init_len)]


def main():
    notch_count=12
    max_iter = 2000
    og = OptimalGeometry(notch_count=notch_count, init_len=256,max_iter=max_iter)
    print(og.notch_bounds())
    lengths = og.init_len_list

    for length in lengths:
        print(length, -og.gamma(lengths=length))

    processors = os.cpu_count()
    print("Available Processors: {}".format(processors))

    args = og.gen_args()
    p = Pool(processes=processors)
    result = p.map(og.minimize, args)
    print(notch_count, sorted(list(result), reverse=True))


if __name__ == '__main__':
    main()
