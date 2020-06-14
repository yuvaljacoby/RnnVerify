import pickle
import sys
from functools import partial

import numpy as np
from tqdm import tqdm

from RNN.Adversarial import adversarial_query, get_out_idx
from polyhedron_algorithms.GurobiBased.MultiLayerBase import GurobiMultiLayer
from polyhedron_algorithms.GurobiBased.GurobiPolyhedronRandomImprove import GurobiMultiLayerRandom
from polyhedron_algorithms.GurobiBased.GurobiPolyhedronIISBased import GurobiMultiLayerIIS
from rnn_experiment.self_compare.generate_random_points import POINTS_PATH


def find_first_point(net_path, points_path, other_idx_method, gurobi_ptr_1, gurobi_ptr_2, max_n_iter=10):
    points = pickle.load(open(points_path, "rb"))
    i = 0
    start_point = 3
    pbar = tqdm(total=(max_n_iter + 1 - start_point) * len(points) * len(other_idx_method))
    for n in range(start_point, max_n_iter + 1):
        for point in points:
            m = 0
            for method in other_idx_method:
                try:
                    idx_max, other_idx = get_out_idx(point, n, net_path, method)
                    res, queries_stats, alpha_history = adversarial_query(point, 0.01, idx_max, other_idx, net_path,
                                                                          gurobi_ptr_1, n)
                    if not res:
                        print("finish n=1 UNSAT")
                        # Polyhedron with dim-1 does not work, check if dim 2 works
                        res, queries_stats, alpha_history = adversarial_query(point, 0.01, idx_max, other_idx, net_path,
                                                                              gurobi_ptr_2, n)
                        if res:
                            print("$" * 100)
                            print(m, n)
                            print(net_path)
                            print(point)
                            print(idx_max, other_idx)
                            print("$" * 100)
                            exit(0)
                    else:
                        print("finish n=1 SAT")
                except:
                    pass
                pbar.update(1)
                i += 1
            m += 1


if __name__ == '__main__':
    other_idx_method = [lambda x: np.argsort(x)[-i] for i in range(2, 7)]
    net_path = sys.argv[1] if len(sys.argv) > 1 else \
        "./FMCAD_EXP/models/model_20classes_rnn4_rnn4_fc32_fc32_fc32_0050.ckpt"
    points_path = POINTS_PATH
    gurobi_ptr_1 = partial(GurobiMultiLayer, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                           use_counter_example=True)
    gurobi_ptr_2 = partial(GurobiMultiLayer, polyhedron_max_dim=2, use_relu=True, add_alpha_constraint=True,
                         use_counter_example=True)
    if len(sys.argv) > 2:
        if sys.argv[2] == 'random':
            gurobi_ptr_1 = partial(GurobiMultiLayerRandom, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                                   use_counter_example=True, max_steps=1)
            gurobi_ptr_2 = partial(GurobiMultiLayerRandom, polyhedron_max_dim=2, use_relu=True, add_alpha_constraint=True,
                                   use_counter_example=True, max_steps=15)
        elif sys.argv[2] == 'base':
            gurobi_ptr_1 = partial(GurobiMultiLayer, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                                   use_counter_example=True)
            gurobi_ptr_2 = partial(GurobiMultiLayer, polyhedron_max_dim=2, use_relu=True, add_alpha_constraint=True,
                                   use_counter_example=True)
        elif sys.argv[2] == 'iis':
            gurobi_ptr_1 = partial(GurobiMultiLayerIIS, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                                   use_counter_example=True, max_steps=1)
            gurobi_ptr_2 = partial(GurobiMultiLayerIIS, polyhedron_max_dim=2, use_relu=True, add_alpha_constraint=True,
                                   use_counter_example=True, max_steps=15)
        else:
            raise KeyError()

    find_first_point(net_path, points_path, other_idx_method, gurobi_ptr_1, gurobi_ptr_2)
