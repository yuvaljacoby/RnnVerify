import os
import tempfile
from functools import partial

import numpy as np
import tensorflow.keras as k

from RNN.Adversarial import get_out_idx, adversarial_query
from polyhedron_algorithms.GurobiBased.MultiLayerBase import GurobiMultiLayer


def run_once(net_path, gurobi_ptr) -> bool:
    point = np.array([1.0])
    n = 3
    method = lambda x: np.argsort(x)[-2]
    idx_max, other_idx = get_out_idx(point, n, net_path, method)
    res, _, _ = adversarial_query(point, 0.01, idx_max, other_idx, net_path, gurobi_ptr, n)
    return res


def check_if_improve(net_path) -> bool:
    gurobi_ptr1 = partial(GurobiMultiLayer, polyhedron_max_dim=1, use_relu=True, add_alpha_constraint=True,
                          use_counter_example=True)
    gurobi_ptr2 = partial(GurobiMultiLayer, polyhedron_max_dim=2, use_relu=True, add_alpha_constraint=True,
                          use_counter_example=True)

    if not run_once(net_path, gurobi_ptr1):
        print('n=1: UNSAT')
        if run_once(net_path, gurobi_ptr2):
            return True
        else:
            print('n=2: UNSAT')
            exit(0)
    return False


def run_random_exp():
    with tempfile.TemporaryFile() as fp:
        file_path = "{}.h5".format(fp.name)
        model = k.Sequential()
        model.add(k.layers.SimpleRNN(2, input_shape=(None, 1), activation='relu', return_sequences=False))
        model.add(k.layers.Dense(2, activation='relu'))
        r1 = np.random.rand(2) + 0.2
        r22 = np.random.rand(1)[0] * -1
        w_h = np.array([r1, [1.0, r22]])

        model.layers[0].set_weights([np.array([1.0, 1.0])[None, :], w_h, np.array([0., 0.])])
        # model.layers[0].set_weights([np.array([1.0,1.0])[None, :], np.array([[1.0,1.0],[1.0,1.0]]), np.array([0.,0.])])
        model.layers[1].set_weights([np.array([[2.0, 0], [0, 1.0]]), np.array([0., 0.])])
        if not os.path.exists(fp.name):
            raise Exception
        model.save(file_path)

        print("$" * 100)
        print("w_h: {}".format(w_h))
        print("$" * 100)
        if check_if_improve(file_path):
            print("*" * 100)
            print("FOUND")
            print("w_h: {}".format(w_h))
            print("*" * 100)
            exit(0)


if __name__ == '__main__':
    for _ in range(1000):
        run_random_exp()
