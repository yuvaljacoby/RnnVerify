from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from RNN.MarabouRNNMultiDim import prove_multidim_property, negate_equation
from RNN.MarabouRnnModel import RnnMarabouModel
from maraboupy import MarabouCore


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


def get_out_idx(x, n_iterations, h5_file_path, other_index_func=lambda vec: np.argmin(vec)):
    # TODO:Change name to get_out_idx_keras
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

    start_initialize_query = timer()
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
    end_initialize_query = timer()

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
        queries_stats['query_initialize'] = end_initialize_query - start_initialize_query

    # if 'invariant_queries' in queries_stats and 'property_queries' in queries_stats and \
    #         queries_stats['property_queries'] != queries_stats['invariant_queries']:
    #     print("What happened?\n", x)
    del rnn_model
    return res, queries_stats, algorithm.alpha_history
