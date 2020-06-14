import os
import pickle
from functools import partial
from timeit import default_timer as timer
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf

from maraboupy import MarabouCore
from RNN.MarabouRNNMultiDim import negate_equation, prove_multidim_property
from RNN.MarabouRnnModel import RnnMarabouModel
from rnn_algorithms.IterateAlphasSGD import IterateAlphasSGD
from rnn_algorithms.Update_Strategy import Absolute_Step, Relative_Step
from rnn_algorithms.GurobiBased import AlphasGurobiBased

MODELS_FOLDER = "/home/yuval/projects/Marabou/models/"

WORKING_EXAMPLES_FOLDER = "/home/yuval/projects/Marabou/working_arrays"


def calc_min_max_template_by_radius(x: list, radius: float):
    '''
    Calculate bounds of first iteration when the input vector is by template (i == 0)
    :param x: list of tuples, each tuple is (alpha, beta) which will be used as alpha * i + beta
    :param radius: non negative number, l_infinity around each of the points
    :return: xlim -  list of tuples each tuple is (lower_bound, upper_bound)
    '''
    assert radius >= 0
    xlim = []
    for (alpha, beta) in x:
        if beta != 0:
            xlim.append((beta * (1 - radius), beta * (1 + radius)))
        else:
            xlim.append((-radius, radius))
    return xlim


def calc_min_max_by_radius(x, radius):
    '''
    :param x: base_vector (input vector that we want to find a ball around it), need to be valid shape for the model
    :param radius: determines the limit of the inputs around the base_vector, non negative number
    :return: xlim -  list of tuples each tuple is (lower_bound, upper_bound)
    '''
    assert radius >= 0
    xlim = []
    for val in x:
        if val > 0:
            xlim.append((val * (1 - radius), val * (1 + radius)))
        elif val < 0:
            xlim.append((val * (1 + radius), val * (1 - radius)))
        else:
            xlim.append((-radius, radius))
    return xlim


def assert_adversarial_query_wrapper(x, y_idx_max, other_idx, h5_file_path, n_iterations, is_fail_test=False):
    '''
    Make sure that when running the network with input x the output at index y_idx_max is larger then other_idx
    if_fail_test negates this in order to let failing tests run
    :param x: 3d input vector where (1, n_iteration, input_dim), or 1d vector (with only input_dim), or list of input values
    :param y_idx_max: max index in the out vector
    :param other_idx: other index in the out vector
    :param h5_file_path: path to h5 file with the network
    :param n_iterations:  number of iterations. if x is a 3d vector ignoring
    :param is_fail_test: to negate the asseration or not
    :return: assert that predict(x)[y_idx_max] >= predict(x)[other_idx]
    '''
    model = tf.keras.models.load_model(h5_file_path)
    if type(x) == list:
        x = np.array(x)
    if len(x.shape) == 1:
        x = np.repeat(x[None, None, :], n_iterations, axis=1)
        # x = np.repeat(x, n_iterations).reshape(1, n_iterations, -1)

    out_vec = np.squeeze(model.predict(x))
    res = out_vec[y_idx_max] >= out_vec[other_idx]
    res = not res if is_fail_test else res
    assert res, out_vec


def assert_adversarial_query_wrapper_template(x, y_idx_max, other_idx, h5_file_path, n_iterations, is_fail_test=False):
    '''
    Make sure that when running the network with input x the output at index y_idx_max is larger then other_idx
    if_fail_test negates this in order to let failing tests run
    :param x: 3d input vector where (1, n_iteration, input_dim), or 1d vector (with only input_dim), or list of input values
    :param y_idx_max: max index in the out vector
    :param other_idx: other index in the out vector
    :param h5_file_path: path to h5 file with the network
    :param n_iterations:  number of iterations. if x is a 3d vector ignoring
    :param is_fail_test: to negate the asseration or not
    :return: assert that predict(x)[y_idx_max] >= predict(x)[other_idx]
    '''
    model = tf.keras.models.load_model(h5_file_path)
    if type(x) == list:
        x = template_to_vector(x, n_iterations)

    out_vec = np.squeeze(model.predict(x))
    res = out_vec[y_idx_max] >= out_vec[other_idx]
    res = not res if is_fail_test else res
    assert res, out_vec


def get_output_vector(h5_file_path: str, x: list, n_iterations: int):
    '''
    predict on the model using x (i.e. the matrix will be of shape (1, n_iterations, len(x))
    :param h5_file_path: path to the model file
    :param x: list of values
    :param n_iterations: number of iteration to create
    :return: ndarray, shape is according to the model output
    '''
    model = tf.keras.models.load_model(h5_file_path)
    tensor = np.repeat(np.array(x)[None, None, :], n_iterations, axis=1)
    return model.predict(tensor)


def template_to_vector(x: list, n_iterations: int):
    '''
    Create a np.array dimension [1, n_iterations, len(x)] according to the template in x
    :param x: list of tuples, where a tuple is (alpha, beta) and the value will be alpha * i + beta (where i is time)
    :param n_iterations:
    :return: np.array dim: [1, n_iterations, len(x)]
    '''

    beta = np.array([t[1] for t in x])
    alpha = np.array([t[0] for t in x])
    tensor = np.zeros(shape=(1, len(x))) + beta
    for i in range(1, n_iterations):
        vec = alpha * i + beta
        tensor = np.vstack((tensor, vec))
    return tensor[None, :, :]


def get_output_vector_template(h5_file_path: str, x: list, n_iterations: int):
    model = tf.keras.models.load_model(h5_file_path)
    tensor = template_to_vector(x, n_iterations)
    return model.predict(tensor)


def get_out_idx(x, n_iterations, h5_file_path, other_index_func=lambda vec: np.argmin(vec)):
    '''
    Calcuate the output vector of h5_file for n_iterations repetations of x vector
    :param x: input vector
    :param n_iterations: how many times to repeat x
    :param h5_file_path: model
    :param other_index_func: function to pointer that gets the out vector (1d) and returns the other inedx
    :return: max_idx, other_idx (None if they are the same)
    '''
    out = np.squeeze(get_output_vector(h5_file_path, x, n_iterations))
    other_idx = other_index_func(out)  # np.argsort(out)[-2]
    y_idx_max = np.argmax(out)
    # assert np.argmax(out) == np.argsort(out)[-1]
    # print(y_idx_max, other_idx)
    if y_idx_max == other_idx:
        # This means all the enteris in the out vector are equal...
        return None, None
    return y_idx_max, other_idx


class Predicate:
    def __init__(self, vars_coefficients: List[Tuple[int, float]], scalar: float, on_input: bool,
                 eq_type=MarabouCore.Equation.GE):
        self.vars_coefficients = vars_coefficients
        self.scalar = scalar
        self.on_input = on_input
        self.eq_type = eq_type

    def get_equation(self, rnn_model: RnnMarabouModel) -> MarabouCore.Equation:
        eq = MarabouCore.Equation(self.eq_type)
        for (v, c) in self.vars_coefficients:
            if self.on_input:
                eq.addAddend(c, rnn_model.input_idx[v])
            else:
                eq.addAddend(c, rnn_model.output_idx[v])
            eq.setScalar(self.scalar)
        return eq


def add_predicates(rnn_model: RnnMarabouModel, P: Optional[List[Predicate]], n_iterations: int) -> None:
    '''
    Adds all the predicates and limits the i variable to be maximum n_iterations
    '''
    if P is None:
        P = []
    P = [p.get_equation() for p in P]

    time_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    time_eq.addAddend(1, rnn_model.get_start_end_idxs(0)[0][0])
    time_eq.setScalar(n_iterations)
    P.append(time_eq)
    for con in P:
        rnn_model.network.addEquation(con)


def query(xlim: List[Tuple[float, float]], P: Optional[List[Predicate]], Q: List[Predicate], h5_file_path: str,
          algorithm_ptr, n_iterations=10, steps_num=5000):
    '''
    :param xlim:  list of tuples each tuple is (lower_bound, upper_bound), input bounds
    :param P: predicates on the input (linear constraints), besides the bounds on the input (xlim)
    :param Q: conditions on the output (linear constraints), not negated
    :param h5_file_path: path to keras model which we will check on
    :param n_iterations: number of iterations to run
    :return: True / False, and queries_stats
    '''
    rnn_model = RnnMarabouModel(h5_file_path, n_iterations)
    rnn_model.set_input_bounds(xlim)

    add_predicates(rnn_model, P, n_iterations)

    start_initial_alg = timer()
    algorithm = algorithm_ptr(rnn_model, xlim)
    end_initial_alg = timer()
    Q_negate = []
    for q in Q:
        Q_negate.append(negate_equation(q.get_equation(rnn_model)))

    res, queries_stats = prove_multidim_property(rnn_model, Q_negate, algorithm, debug=1, return_queries_stats=True,
                                                 number_of_steps=steps_num)

    if queries_stats:
        step_times = queries_stats['step_times']['raw']
        step_times.insert(0, end_initial_alg - start_initial_alg)
        queries_stats['step_times'] = {'avg': np.mean(step_times), 'median': np.median(step_times), 'raw': step_times}
        queries_stats['step_queries'] = len(step_times)

    return res, queries_stats, algorithm.alpha_history


def adversarial_query(x: list, radius: float, y_idx_max: int, other_idx: int, h5_file_path: str, algorithm_ptr,
                      n_iterations=10, steps_num=5000):
    '''
    Query marabou with adversarial query
    :param x: base_vector (input vector that we want to find a ball around it)
    :param radius: determines the limit of the inputs around the base_vector
    :param y_idx_max: max index in the output layer
    :param other_idx: which index to compare max idx
    :param h5_file_path: path to keras model which we will check on
    :param algorithm_ptr: TODO
    :param n_iterations: number of iterations to run
    :return: True / False, and queries_stats
    '''

    if y_idx_max is None or other_idx is None:
        y_idx_max, other_idx = get_out_idx(x, n_iterations, h5_file_path)
        if y_idx_max == other_idx or y_idx_max is None or other_idx is None:
            # This means all the enteris in the out vector are equal...
            return False, None, None

    xlim = calc_min_max_by_radius(x, radius)
    rnn_model = RnnMarabouModel(h5_file_path, n_iterations)
    rnn_model.set_input_bounds(xlim)

    # output[y_idx_max] >= output[0] <-> output[y_idx_max] - output[0] >= 0, before feeding marabou we negate this
    adv_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    adv_eq.addAddend(-1, rnn_model.output_idx[other_idx])
    adv_eq.addAddend(1, rnn_model.output_idx[y_idx_max])
    adv_eq.setScalar(0)

    time_eq = MarabouCore.Equation()
    time_eq.addAddend(1, rnn_model.get_start_end_idxs(0)[0][0])
    time_eq.setScalar(n_iterations)

    start_initial_alg = timer()
    algorithm = algorithm_ptr(rnn_model, xlim)
    end_initial_alg = timer()
    # rnn_model.network.dump()

    res, queries_stats = prove_multidim_property(rnn_model, [negate_equation(adv_eq), time_eq], algorithm, debug=1,
                                                 return_queries_stats=True, number_of_steps=steps_num)
    if queries_stats:
        step_times = queries_stats['step_times']['raw']
        step_times.insert(0, end_initial_alg - start_initial_alg)
        queries_stats['step_times'] = {'avg': np.mean(step_times), 'median': np.median(step_times), 'raw': step_times}
        queries_stats['step_queries'] = len(step_times)

    if 'invariant_queries' in queries_stats and 'property_queries' in queries_stats and \
            queries_stats['property_queries'] != queries_stats['invariant_queries']:
        print("What happened?\n", x)
    return res, queries_stats, algorithm.alpha_history


def adversarial_query_wrapper(x: list, radius: float, y_idx_max: int, other_idx: int, h5_file_path: str,
                              n_iterations=10,
                              is_fail_test=False, alpha_step_policy_ptr=Absolute_Step):
    '''
    Query marabou with adversarial query
    :param x: base_vector (input vector that we want to find a ball around it)
    :param radius: determines the limit of the inputs around the base_vector
    :param y_idx_max: max index in the output layer, if None run the model with x for n_iterations and extract it
    :param other_idx: which index to compare max idx, if None the minimum when running the model with x for n iterations
    :param h5_file_path: path to keras model which we will check on
    :param n_iterations: number of iterations to run
    :param is_fail_test: we make sure that the output in y_idx_max >= other_idx, if this is True we negate it
    :return: True / False
    '''
    # TODO: Remove from here, decide waht to do with the assert
    if y_idx_max is None or other_idx is None:
        out = get_output_vector(h5_file_path, x, n_iterations)
        if other_idx is None:
            other_idx = np.argmin(out)
        if y_idx_max is None:
            y_idx_max = np.argmax(out)
        print(y_idx_max, other_idx)

    assert_adversarial_query_wrapper(x, y_idx_max, other_idx, h5_file_path, n_iterations, is_fail_test)
    rnn_model = RnnMarabouModel(h5_file_path, n_iterations)
    xlim = calc_min_max_by_radius(x, radius)
    rnn_model.set_input_bounds(xlim)

    # output[y_idx_max] >= output[0] <-> output[y_idx_max] - output[0] >= 0, before feeding marabou we negate this
    adv_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    adv_eq.addAddend(-1, rnn_model.output_idx[other_idx])  # The zero'th element in the output layer
    adv_eq.addAddend(1, rnn_model.output_idx[y_idx_max])  # The y_idx_max of the output layer
    # adv_eq
    # .addAddend(-1, 3)  # The y_idx_max of the output layer
    adv_eq.setScalar(0)

    algorithm_ptr = partial(IterateAlphasSGD, update_strategy_ptr=alpha_step_policy_ptr)
    res, _, _ = adversarial_query(x, radius, y_idx_max, other_idx, h5_file_path, algorithm_ptr)
    # res, num_queries = prove_multidim_property(rnn_model, [negate_equation(adv_eq)], algorithm, return_alphas=True)
    return res


def test_20classes_1rnn2_1fc2_fail():
    n_inputs = 40
    y_idx_max = 10
    other_idx = 19

    assert not adversarial_query_wrapper([79] * n_inputs, 0, y_idx_max, other_idx,
                                         "{}/model_classes20_1rnn2_1_2_4.h5".format(MODELS_FOLDER), is_fail_test=True)


def test_20classes_1rnn2_1fc2_pass():
    n_inputs = 40
    y_idx_max = 19
    other_idx = 10

    assert adversarial_query_wrapper([79] * n_inputs, 0, y_idx_max, other_idx,
                                     "{}/model_classes20_1rnn2_1_2_4.h5".format(MODELS_FOLDER), is_fail_test=False)


def test_20classes_1rnn2_1fc32_pass():
    n_inputs = 40
    y_idx_max = 13
    other_idx = 19

    results = []
    for i in range(20):
        if other_idx != y_idx_max:
            other_idx = i
            results.append(adversarial_query_wrapper([1] * n_inputs, 0, y_idx_max, other_idx,
                                                     "{}/model_classes20_1rnn2_1_32_4.h5".format(MODELS_FOLDER),
                                                     is_fail_test=False))
            print(results)
    assert sum(results) == 19, 'managed to prove on {}%'.fromat((19 - sum(results)) / 19)


def test_20classes_1rnn2_1fc32_fail():
    n_inputs = 40
    y_idx_max = 0
    other_idx = 13

    assert not adversarial_query_wrapper([1] * n_inputs, 0.1, y_idx_max, other_idx,
                                         "{}/model_classes20_1rnn2_1_32_4.h5".format(MODELS_FOLDER),
                                         is_fail_test=True)


def test_20classes_1rnn2_1fc32_pass():
    n_inputs = 40
    y_idx_max = 13
    other_idx = 0

    assert adversarial_query_wrapper([1] * n_inputs, 0.05, y_idx_max, other_idx,
                                     "{}/model_classes20_1rnn2_1_32_4.h5".format(MODELS_FOLDER),
                                     is_fail_test=False)


def test_20classes_1rnn2_0fc_pass():
    n_iterations = 1000
    n_inputs = 40
    # output[0] < output[1] so this should fail to prove
    y_idx_max = 9
    other_idx = 2
    assert adversarial_query_wrapper([10] * n_inputs, 0.1, y_idx_max, other_idx,
                                     "{}/model_classes20_1rnn2_0_64_4.h5".format(MODELS_FOLDER),
                                     n_iterations=n_iterations)
    return
    # results = []
    # for i in range(20):
    #     if i != y_idx_max:
    #         other_idx = i
    #         results.append(adversarial_query_wrapper([10] * n_inputs, 0.1, y_idx_max, other_idx,
    #                                          "{}/model_classes20_1rnn2_0_64_4.h5".format(MODELS_FOLDER)))
    # print(results)
    # assert sum(results) == 19, 'managed to prove on: {}%'.format((19 - sum(results)) / 19)


def test_20classes_1rnn2_0fc_fail():
    n_inputs = 40
    y_idx_max = 2
    other_idx = 9
    # 6.199209
    assert not adversarial_query_wrapper([10] * n_inputs, 0.1, y_idx_max, other_idx,
                                         "{}/model_classes20_1rnn2_0_64_4.h5".format(MODELS_FOLDER), is_fail_test=True)


def test_20classes_1rnn3_0fc_pass():
    n_inputs = 40
    # output[0] < output[1] so this should fail to prove
    y_idx_max = 10
    other_idx = 19
    assert adversarial_query_wrapper([10] * n_inputs, 0.001, y_idx_max, other_idx,
                                     "{}/model_classes20_1rnn3_0_2_4.h5".format(MODELS_FOLDER),
                                     n_iterations=5)
    return


def test_20classes_1rnn3_0fc_fail():
    n_inputs = 40
    y_idx_max = 19
    other_idx = 10
    # 6.199209
    assert not adversarial_query_wrapper([10] * n_inputs, 0.1, y_idx_max, other_idx,
                                         "{}/model_classes20_1rnn3_0_2_4.h5".format(MODELS_FOLDER), is_fail_test=True,
                                         n_iterations=5)


def test_20classes_1rnn4_0fc_pass():
    n_inputs = 40
    y_idx_max = 13  # 8
    other_idx = 15  # 12
    in_tensor = np.array([6.3, 9.4, 9.6, 3.1, 8.5, 9.4, 7.2, 8.6, 3.8, 1.4, 0.7, 7.8, 1.9, 8.2, 6.2, 3.6, 8.7, 1.7
                             , 2.8, 4.8, 4.3, 5.1, 3.8, 0.8, 2.4, 7.6, 7.3, 0., 3.3, 7.4, 0., 2.1, 0.5, 8., 7.1, 3.9
                             , 3., 8.3, 5.6, 1.8])
    assert in_tensor.shape[0] == n_inputs
    assert adversarial_query_wrapper(in_tensor, 0.01, y_idx_max, other_idx,
                                     "{}/model_classes20_1rnn4_0_2_4.h5".format(MODELS_FOLDER), n_iterations=5)


def test_20classes_1rnn4_0fc_fail():
    n_inputs = 40
    y_idx_max = 15
    other_idx = 13
    in_tensor = np.array([6.3, 9.4, 9.6, 3.1, 8.5, 9.4, 7.2, 8.6, 3.8, 1.4, 0.7, 7.8, 1.9, 8.2, 6.2, 3.6, 8.7, 1.7
                             , 2.8, 4.8, 4.3, 5.1, 3.8, 0.8, 2.4, 7.6, 7.3, 0., 3.3, 7.4, 0., 2.1, 0.5, 8., 7.1, 3.9
                             , 3., 8.3, 5.6, 1.8])
    assert not adversarial_query_wrapper(in_tensor, 0, y_idx_max, other_idx,
                                         "{}/model_classes20_1rnn4_0_64_4.h5".format(MODELS_FOLDER), is_fail_test=True)


def test_20classes_1rnn8_0fc():
    n_inputs = 40
    y_idx_max = 1
    other_idx = 0
    in_tensor = [0.19005403, 0.51136299, 0.67302099, 0.59573087, 0.78725824,
                 0.47257432, 0.65504724, 0.69202802, 0.16531246, 0.84543712,
                 0.73715671, 0.03674481, 0.39459927, 0.0107714, 0.15337461,
                 0.44855902, 0.894079, 0.48551109, 0.08504609, 0.74320624,
                 0.52363974, 0.80471539, 0.06424345, 0.65279486, 0.15554268,
                 0.63541206, 0.15977761, 0.70137553, 0.34406331, 0.59930546,
                 0.8740703, 0.89584981, 0.67799938, 0.78253788, 0.33091662,
                 0.74464927, 0.69366703, 0.96878231, 0.58014617, 0.41094702]
    n_iterations = 5
    assert adversarial_query_wrapper(in_tensor, 0, y_idx_max, other_idx,
                                     "{}/model_classes20_1rnn8_0_64_100.h5".format(MODELS_FOLDER),
                                     n_iterations=n_iterations)

    assert not adversarial_query_wrapper(in_tensor, 0, other_idx, y_idx_max,
                                         "{}/model_classes20_1rnn8_0_64_100.h5".format(MODELS_FOLDER),
                                         n_iterations=n_iterations)


def test_20classes_1rnn4_1fc32():
    n_inputs = 40
    y_idx_max = 9
    other_idx = 14
    in_tensor = np.array([0.43679032, 0.51105192, 0.01603254, 0.45879329, 0.64639347,
                          0.39209051, 0.98618169, 0.49293316, 0.70440262, 0.08594672,
                          0.17252591, 0.4940284, 0.83947774, 0.55545332, 0.8971317,
                          0.72996308, 0.23706766, 0.66869303, 0.74949942, 0.57524252,
                          0.94886307, 0.31034989, 0.41785656, 0.5697128, 0.74751913,
                          0.48868271, 0.22672374, 0.6350584, 0.88979192, 0.97493685,
                          0.96969836, 0.99457045, 0.89433312, 0.19916606, 0.63957592,
                          0.02826659, 0.08104817, 0.20176526, 0.1114994, 0.29297289])
    assert in_tensor.shape[0] == n_inputs
    assert adversarial_query_wrapper(in_tensor, 0.01, y_idx_max, other_idx,
                                     "{}/model_classes20_1rnn4_1_32_4.h5".format(MODELS_FOLDER), n_iterations=10)

    assert not adversarial_query_wrapper(in_tensor, 0.01, other_idx, y_idx_max,
                                         "{}/model_classes20_1rnn4_1_32_4.h5".format(MODELS_FOLDER), n_iterations=10,
                                         is_fail_test=True)


def test_20classes_1rnn8_1fc32():
    n_inputs = 40
    y_idx_max = 14
    other_idx = 3
    in_tensor = np.array([0.90679393, 0.90352916, 0.1756208, 0.99622917, 0.31828876,
                          0.54578732, 0.15096196, 0.19435984, 0.58806244, 0.46534135, 0.82525653, 0.61739753,
                          0.47004321, 0.66255417, 0.78319261, 0.68970699, 0.50609439, 0.68917296, 0.87666094, 0.8222427,
                          0.10933717, 0.86577764, 0.90037717, 0.85837105, 0.30076024, 0.31086682, 0.24680442,
                          0.95077129, 0.44299597, 0.98173942, 0.95088949, 0.24104634, 0.25912628, 0.72127712, 0.8212451,
                          0.50530752, 0.84822531, 0.87344498, 0.60873922, 0.69857207]
                         )

    assert in_tensor.shape[0] == n_inputs
    assert adversarial_query_wrapper(in_tensor, 0.01, y_idx_max, other_idx,
                                     "{}/model_classes20_1rnn8_1_32_4.h5".format(MODELS_FOLDER), n_iterations=10)

    assert not adversarial_query_wrapper(in_tensor, 0.01, other_idx, y_idx_max,
                                         "{}/model_classes20_1rnn8_1_32_4.h5".format(MODELS_FOLDER), n_iterations=10,
                                         is_fail_test=True)


def search_for_input(path, algorithm_ptr, radius=0.01, mean=3, var=0.5):
    n_inputs = 40
    # not_found = True
    n_iterations = 50
    # TODO: If in_tensor has a negative compponnet everything breaks, not sure why :(
    in_tensor = np.random.normal(mean, var, (n_inputs,))
    # print(in_tensor)
    # in_tensor = np.random.random((n_inputs, ))

    res, num_iterations, _ = adversarial_query(in_tensor, radius, None, None, path, algorithm_ptr, n_iterations,
                                               steps_num=1500)
    print("finish one query, res: {}, iterations: {}".format(res, num_iterations))

    if res:
        # found an example that works, try to get multiple adv queries that work
        out = get_output_vector(path, in_tensor, n_iterations)

        other_idx = np.argmin(out)
        y_idx_max = np.argmax(out)

        return {'in_tensor': in_tensor, 'n_iterations': n_iterations, 'idx_max': y_idx_max,
                'other_idx': other_idx, 'radius': radius}
    else:
        return None


def search_for_input_multiple(path, algorithm_ptr, radius=0.01, mean=10, var=3):
    '''
    Searching for inputs that we can prove advarserial robustness on them
    sampling from a normal distribution input points until proving
    :param path: path to h5 file with the model
    :param algorithm_ptr:  huerstics for the rnn invariant search
    :param radius: radius of the ball around the input point
    :param mean: normal distribution mean
    :param var: normal distribution variance
    :return:
    '''

    examples_found = 0

    net_name = path.split(".")[0].split("/")[-1]
    pickle_path = "{}/{}2.pkl".format(WORKING_EXAMPLES_FOLDER, net_name)
    if os.path.exists(pickle_path):
        examples = pickle.load(open(pickle_path, "rb"))
    else:
        examples = []
    print("Searching for inputs on network: {}".format(path))
    for j in range(300):
        start = timer()
        example = search_for_input(path, algorithm_ptr, radius, mean, var)
        if example is not None:
            examples.append(example)
            with open(pickle_path, "wb") as f:
                pickle.dump(examples, f)

            print("###### found example {}: {} ### \n {} \n ############".format(examples_found, net_name,
                                                                                 str(example['in_tensor']).replace(' ',
                                                                                                                   ', ')))
            examples_found += 1
            # for i in range(len(out[0])):
            #     if i != other_idx and i != y_idx_max:
            #         res, num_iterations, _ = adversarial_query(in_tensor, radius, y_idx_max, i, path, algorithm_ptr,
            #                                                    n_iterations, steps_num=1500)
            #         if res:
            #             examples.append({'in_tensor': in_tensor, 'n_iterations': n_iterations, 'idx_max': y_idx_max,
            #                              'other_idx': i, 'radius': radius})
            #
            #             print("###### found example {}: {} ### \n {} \n ############".format(examples_found,
            #                                                                                  net_name,
            #                                                                                  str(in_tensor)
            #                                                                                  .replace(' ', ', ')))
            #             examples_found += 1

            # with open(pickle_path, "wb") as f:
            #     pickle.dump(examples, f)
            end = timer()
            # print("*************** fail to prove iteration: {} took: {} ***************".format(j, end - start))


if __name__ == "__main__":
    exp = {'idx_max': 1, 'other_idx': 4,
           'in_tensor': np.array([0.23300637, 0.0577466, 0.88960908, 0.02926062, 0.4322654,
                                  0.05116153, 0.93342266, 0.3143915, 0.39245229, 0.1144419,
                                  0.08748452, 0.24332963, 0.34622415, 0.42573235, 0.26952168,
                                  0.53801347, 0.26718764, 0.24274057, 0.11475819, 0.9423371,
                                  0.70257952, 0.34443971, 0.08917664, 0.50140514, 0.75890139,
                                  0.65532994, 0.74165648, 0.46543468, 0.00583174, 0.54016713,
                                  0.74460554, 0.45771724, 0.59844178, 0.73369685, 0.50576504,
                                  0.91561612, 0.39746448, 0.14791963, 0.38114261, 0.24696231]),
           'radius': 0, 'h5_path': "{}/old/model_classes5_1rnn2_0_64_4.h5".format(MODELS_FOLDER), 'n_iterations': 5}

    rnn_model = RnnMarabouModel("{}/old/model_classes5_1rnn2_0_64_4.h5".format(MODELS_FOLDER), 50)
    w_in, w_h, b = rnn_model.get_weights()[0]
    xlim = calc_min_max_by_radius(exp['in_tensor'], exp['radius'])
    initial_values = rnn_model.get_rnn_min_max_value_one_iteration(xlim)
    # build_gurobi_query(initial_values[1], w_in, w_h, b, exp['n_iterations'])

    # search_for_input("/home/yuval/projects/Marabou/models_new_vctk/model_classes20_1rnn3_1_32_3.h5", relative_step)
    # in_tensor = np.array([0.37205514, 0.84609851, 0.34888652, 0.099101, 0.8797378, 0.02679134
    #                          , 0.18232116, 0.18231391, 0.12444646, 0.8643345, 0.77595206, 0.16838746
    #                          , 0.22769657, 0.55295006, 0.32333069, 0.26841413, 0.67070145, 0.96513381
    #                          , 0.89063008, 0.11651877, 0.30640328, 0.70550923, 0.01069241, 0.22659354
    #                          , 0.11761449, 0.35928134, 0.13414231, 0.56152431, 0.34172535, 0.81053337
    #                          , 0.37676732, 0.19970681, 0.60641318, 0.20872408, 0.20356423, 0.24063641
    #                          , 0.32073923, 0.41748575, 0.44155234, 0.63568076])
    # assert adversarial_query_wrapper(in_tensor, 0, None, None,
    #                          "/home/yuval/projects/Marabou/models_new_vctk/model_classes20_1rnn3_1_32_3.h5",
    #                          is_fail_test=False, alpha_step_policy_ptr=Relative_Step)
    #
    # exit(0)
    # search_for_input("{}/model_classes20_1rnn4_0_2_4.h5".format(MODELS_FOLDER))
    # IterateAlphasSGD_absolute_step = partial(IterateAlphasSGD, update_strategy_ptr=Absolute_Step)
    gurobi_relative_step = partial(AlphasGurobiBased, update_strategy_ptr=Relative_Step)

    res, iterations, alpha_history = adversarial_query(exp['in_tensor'], exp['radius'], exp['idx_max'],
                                                       exp['other_idx'], exp['h5_path'],
                                                       gurobi_relative_step, exp['n_iterations'], steps_num=1000)
    #
    # max_alphas_history = [a[-2:] for a in alpha_history]
    # from maraboupy.draw_rnn import draw_2d_from_h5
    #
    # draw_2d_from_h5(exp['h5_path'], exp['in_tensor'], exp['n_iterations'], max_alphas_history)
    # exit(0)

    # NOTE TO YUVAL - When running this, disabled the assert inside property_oracle to make things run faster
    # search_for_input_multiple("{}/model_20classes_rnn4_fc16_fc32_epochs3.h5".format(MODELS_FOLDER), weighted_relative_step)
    # search_for_input_multiple("{}/model_20classes_rnn4_fc32_epochs40.h5".format(MODELS_FOLDER), weighted_relative_step)
    #

    #
    # import multiprocessing
    # from functools import partial
    # worker = partial(search_for_input, path="{}/model_classes20_1rnn8_1_32_4.h5".format(MODELS_FOLDER))
    # for i in range(5):
    #     p = multiprocessing.Process(target=worker)
    #     p.start()
    # search_for_input("{}/model_classes20_1rnn8_0_32_4.h5".format(MODELS_FOLDER))
    # search_for_input("{}/model_classes20_1rnn8_1_32_4.h5".format(MODELS_FOLDER))

    # test_20classes_1rnn2_0fc_pass()
    # test_20classes_1rnn2_0fc_fail()
    # test_20classes_1rnn2_0fc_template_input_pass()
    # test_20classes_1rnn2_0fc_template_input_fail()
    # test_20classes_1rnn2_1fc32_pass()
    # test_20classes_1rnn2_1fc32_fail()
    #
    # test_20classes_1rnn3_0fc_pass()
    # test_20classes_1rnn3_0fc_fail()
    # test_20classes_1rnn4_0fc_pass()
    # test_20classes_1rnn4_0fc_fail()
    # test_20classes_1rnn4_1fc32()
    #
    # test_20classes_1rnn8_0fc()
