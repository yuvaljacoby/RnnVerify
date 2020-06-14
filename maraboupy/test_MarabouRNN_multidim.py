from maraboupy import MarabouCore
from RNN.MarabouRNNMultiDim import add_rnn_multidim_cells, \
    negate_equation, prove_multidim_property
from rnn_algorithms.IterateAlphasSGD import IterateAlphasSGD

import numpy as np

LARGE = 50000.0
SMALL = 10 ** -2


def relu(num):
    return max(0, num)


def define_adversarial_robustness_two_input_nodes_two_hidden(xlim, n_iterations):
    '''
    Define an adversarial roubstness examples
    0 <= x_0 <= 1
    1 <= x_1 <= 2
    s_i = 1 * x_0 + 5 * x_1 + 1 * s_(i-1)
    z_i = 2 * x_0 + 1 * x_1 + 1 * z_(i-1)
    A = s_i
    B = z_i
    therefore:
        5 <= A <= 6
        1 <= B <= 4
    prove that after n_iterations A >= B
    :param xlim: array of tuples, each array cell as an input (x_0, x_1 etc..) tuple is (min_value, max_value)
    :param n_iterations: number of iterations
    :return: network, rnn_output_idx, initial_values, adv_eq
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(len(xlim))  # x1, x2

    # x1
    network.setLowerBound(0, xlim[0][0])
    network.setUpperBound(0, xlim[0][1])

    # x2
    network.setLowerBound(1, xlim[1][0])
    network.setUpperBound(1, xlim[1][1])

    s_s_hidden_w = 1
    s_z_hidden_w = 1
    z_s_hidden_w = 0.9
    z_z_hidden_w = 1
    x0_s_w = 1
    x1_s_w = 4
    x0_z_w = 2
    x1_z_w = 1

    rnn_output_idx = add_rnn_multidim_cells(network, [0, 1], np.array([[x0_s_w, x1_s_w], [x0_z_w, x1_z_w]]),
                                            np.array([[s_s_hidden_w, s_z_hidden_w], [z_s_hidden_w, z_z_hidden_w]]),
                                            [0, 0], n_iterations)
    a_idx = rnn_output_idx[-1] + 1
    b_idx = a_idx + 1

    a_w = 1
    b_w = 1

    network.setNumberOfVariables(b_idx + 1)

    # A
    network.setLowerBound(a_idx, -LARGE)  # A
    network.setUpperBound(a_idx, LARGE)

    # B
    network.setLowerBound(b_idx, -LARGE)  # B
    network.setUpperBound(b_idx, LARGE)

    # # A = skf <--> A - skf = 0
    a_output_eq = MarabouCore.Equation()
    a_output_eq.addAddend(1, a_idx)
    a_output_eq.addAddend(-a_w, rnn_output_idx[0])
    a_output_eq.setScalar(0)
    a_output_eq.dump()
    network.addEquation(a_output_eq)

    # B = zkf <--> B - z_k_f = 0
    b_output_eq = MarabouCore.Equation()
    b_output_eq.addAddend(1, b_idx)
    b_output_eq.addAddend(-b_w, rnn_output_idx[1])
    b_output_eq.setScalar(0)
    b_output_eq.dump()
    network.addEquation(b_output_eq)

    min_b = relu(relu(xlim[0][0] * x0_z_w) + relu(xlim[1][0] * x1_z_w) * b_w)
    max_b = relu(relu(xlim[0][1] * x0_z_w) + relu(xlim[1][1] * x1_z_w) * b_w)

    min_a = relu(relu(xlim[0][0] * x0_s_w) + relu(xlim[1][0] * x1_s_w) * a_w)
    max_a = relu(relu(xlim[0][1] * x0_s_w) + relu(xlim[1][1] * x1_s_w) * a_w)
    initial_values = ([min_a, min_b], [max_a, max_b])
    print('a: {} {}'.format(min_a, max_a))
    print('b: {} {}'.format(min_b, max_b))

    # A >0 B <-> A - B >= 0
    adv_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    adv_eq.addAddend(1, a_idx)
    adv_eq.addAddend(-1, b_idx)
    adv_eq.setScalar(0)

    return network, rnn_output_idx, initial_values, [negate_equation(adv_eq)]


def define_adversarial_robustness_two_input_nodes(xlim, n_iterations):
    '''
    Define an adversarial roubstness examples
    0 <= x_0 <= 1
    1 <= x_1 <= 2
    s_i = 1 * x_0 + 5 * x_1 + 1 * s_(i-1)
    z_i = 2 * x_0 + 1 * x_1 + 1 * z_(i-1)
    A = s_i
    B = z_i
    therefore:
        5 <= A <= 6
        1 <= B <= 4
    prove that after n_iterations A >= B
    :param xlim: array of tuples, each array cell as an input (x_0, x_1 etc..) tuple is (min_value, max_value)
    :param n_iterations: number of iterations
    :return: network, rnn_output_idx, initial_values, adv_eq
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(len(xlim))  # x1, x2

    # x1
    network.setLowerBound(0, xlim[0][0])
    network.setUpperBound(0, xlim[0][1])

    # x2
    network.setLowerBound(1, xlim[1][0])
    network.setUpperBound(1, xlim[1][1])

    s_hidden_w = 1
    z_hidden_w = 1
    x0_s_w = 1
    x1_s_w = 4
    x0_z_w = 2
    x1_z_w = 1

    rnn_output_idx = add_rnn_multidim_cells(network, [0, 1], np.array([[x0_s_w, x1_s_w], [x0_z_w, x1_z_w]]),
                                            np.array([[s_hidden_w, 0], [0, z_hidden_w]]),
                                            [0, 0], n_iterations)
    a_idx = rnn_output_idx[-1] + 1
    b_idx = a_idx + 1

    a_w = 1
    b_w = 1

    network.setNumberOfVariables(b_idx + 1)

    # A
    network.setLowerBound(a_idx, -LARGE)  # A
    network.setUpperBound(a_idx, LARGE)

    # B
    network.setLowerBound(b_idx, -LARGE)  # B
    network.setUpperBound(b_idx, LARGE)

    # # A = skf <--> A - skf = 0
    a_output_eq = MarabouCore.Equation()
    a_output_eq.addAddend(1, a_idx)
    a_output_eq.addAddend(-a_w, rnn_output_idx[0])
    a_output_eq.setScalar(0)
    a_output_eq.dump()
    network.addEquation(a_output_eq)

    # B = zkf <--> B - z_k_f = 0
    b_output_eq = MarabouCore.Equation()
    b_output_eq.addAddend(1, b_idx)
    b_output_eq.addAddend(-b_w, rnn_output_idx[1])
    b_output_eq.setScalar(0)
    b_output_eq.dump()
    network.addEquation(b_output_eq)

    min_b = relu(relu(xlim[0][0] * x0_z_w) + relu(xlim[1][0] * x1_z_w) * b_w)
    max_b = relu(relu(xlim[0][1] * x0_z_w) + relu(xlim[1][1] * x1_z_w) * b_w)

    min_a = relu(relu(xlim[0][0] * x0_s_w) + relu(xlim[1][0] * x1_s_w) * a_w)
    max_a = relu(relu(xlim[0][1] * x0_s_w) + relu(xlim[1][1] * x1_s_w) * a_w)
    initial_values = ([min_a, min_b], [max_a, max_b])
    print('a: {} {}'.format(min_a, max_a))
    print('b: {} {}'.format(min_b, max_b))

    # A >0 B <-> A - B >= 0
    adv_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    adv_eq.addAddend(1, a_idx)
    adv_eq.addAddend(-1, b_idx)
    adv_eq.setScalar(0)

    return network, rnn_output_idx, initial_values, [negate_equation(adv_eq)]


def test_adversarial_robustness_two_inputs_SGDAlgorithm():
    '''
    This example has 2 input nodes and two RNN cells
    '''
    num_iterations = 10
    xlim = [(0, 1), (1, 2)]

    network, rnn_output_idxs, initial_values, property_eq = define_adversarial_robustness_two_input_nodes(xlim,
                                                                                                          num_iterations)
    network.dump()
    rnn_start_idxs = [i - 3 for i in rnn_output_idxs]
    algorithm = IterateAlphasSGD(initial_values, rnn_start_idxs, rnn_output_idxs)
    assert prove_multidim_property(network, rnn_start_idxs, rnn_output_idxs, property_eq, algorithm)


def test_adversarial_robustness_two_inputs_SGDAlgorithm_fail():
    '''
    This example has 2 input nodes and two RNN cells
    '''
    num_iterations = 10
    xlim = [(0, 2), (1, 2)]

    network, rnn_output_idxs, initial_values, property_eq = define_adversarial_robustness_two_input_nodes(xlim,
                                                                                                          num_iterations)
    network.dump()
    rnn_start_idxs = [i - 3 for i in rnn_output_idxs]
    algorithm = IterateAlphasSGD(initial_values, rnn_start_idxs, rnn_output_idxs)
    assert not prove_multidim_property(network, rnn_start_idxs, rnn_output_idxs, property_eq, algorithm)


def test_adversarial_robustness_two_inputs_two_hidden_SGDAlgorithm():
    '''
    This example has 2 input nodes and two RNN cells
    '''
    num_iterations = 10
    xlim = [(0, 1), (1, 2)]

    network, rnn_output_idxs, initial_values, property_eq = \
        define_adversarial_robustness_two_input_nodes_two_hidden(xlim, num_iterations)
    network.dump()
    rnn_start_idxs = [i - 3 for i in rnn_output_idxs]
    algorithm = IterateAlphasSGD(initial_values, rnn_start_idxs, rnn_output_idxs)
    assert prove_multidim_property(network, rnn_start_idxs, rnn_output_idxs, property_eq, algorithm)
