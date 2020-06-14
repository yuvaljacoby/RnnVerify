from maraboupy import MarabouCore
from maraboupy.MarabouRNN import *


def define_zero_network(xlim, ylim, n_iterations):
    '''
    Defines the zero network in a marabou way
    The zero network is a network with two rnn cells, that always outputs zero
    :param xlim: how to limit the input to the network
    :param ylim: how to limit the output of the network, will effect how we create the invariant
    :param n_iterations: number of inputs / times the rnn cell will be executed
    :return: query to marabou that defines the positive_sum rnn network (without recurrent)
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(1)  # x

    # x
    network.setLowerBound(0, xlim[0])
    network.setUpperBound(0, xlim[1])

    s_cell_iterator = 1  # i
    s_i_f_idx = add_rnn_cell(network, [(0, 1)], 1, n_iterations)
    z_cell_iterator = network.getNumberOfVariables()
    z_i_f_idx = add_rnn_cell(network, [(0, 1)], 1, n_iterations)
    y_idx = z_i_f_idx + 1

    network.setNumberOfVariables(y_idx + 1)

    # y
    network.setLowerBound(y_idx, -large)
    network.setUpperBound(y_idx, large)

    # y = skf - zkf <--> y - skf + zkf = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_idx)
    output_equation.addAddend(-1, s_i_f_idx)
    output_equation.addAddend(1, z_i_f_idx)
    output_equation.setScalar(0)
    # output_equation.dump()
    network.addEquation(output_equation)

    # s_i f - z_i f <= 0.01
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(-1, z_i_f_idx)  # s_i f
    invariant_equation.addAddend(1, s_i_f_idx)  # s_i f
    invariant_equation.setScalar(SMALL)

    # y <= n * 0.01
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, y_idx)
    property_eq.setScalar(ylim)

    return network, [s_cell_iterator, z_cell_iterator], invariant_equation, [property_eq]


# Simple RNN, only sums the negative inputs
#   0   <= xi  <= 1
#   0   <= sif
#   1/2 <= y  <= 1
#
# Equations:
#  x1 - s1b = 0
#  for each i > 1
#       -xi + s(i-1)f - sib = 0
#  y - skf =0 # Where k == n_iterations
#
#  sif = Relu(sib)
def define_negative_sum_network(xlim, ylim, n_iterations):
    '''
    Defines the negative network in a marabou way
        s_i = ReLu(-1 * x_i + s_i-1)
        y = s_k (where k == n_iterations)
    :param xlim: how to limit the input to the network
    :param ylim: how to limit the output of the network
    :param n_iterations: number of inputs / times the rnn cell will be executed
    :return: query to marabou that defines the positive_sum rnn network (without recurrent)
    '''
    positive_sum_rnn_query = MarabouCore.InputQuery()
    positive_sum_rnn_query.setNumberOfVariables(1)  # x

    # x
    positive_sum_rnn_query.setLowerBound(0, xlim[0])
    positive_sum_rnn_query.setUpperBound(0, xlim[1])

    rnn_start_idx = 1  # i
    rnn_idx = add_rnn_cell(positive_sum_rnn_query, [(0, -1)], 1, n_iterations)  # rnn_idx == s_i f
    y_idx = rnn_idx + 1

    positive_sum_rnn_query.setNumberOfVariables(y_idx + 1)

    # y
    positive_sum_rnn_query.setLowerBound(y_idx, -large)
    positive_sum_rnn_query.setUpperBound(y_idx, large)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_idx)
    output_equation.addAddend(-1, rnn_idx)
    output_equation.setScalar(0)
    # output_equation.dump()
    positive_sum_rnn_query.addEquation(output_equation)

    # s_i f <= i + 1 <--> i - s_i f >= -1
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    invariant_equation.addAddend(1, rnn_start_idx)  # i
    invariant_equation.addAddend(-1, rnn_idx)  # s_i f
    invariant_equation.setScalar(-1)

    # y <= ylim
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, y_idx)
    property_eq.setScalar(ylim[1])

    return positive_sum_rnn_query, [rnn_start_idx], invariant_equation, [property_eq]


# Simple RNN, only sums the positive inputs
#   0   <= xi  <= 1
#   0   <= sif
#   1/2 <= y  <= 1
#
# Equations:
#  x1 - s1b = 0
#  for each i > 1
#       xi + s(i-1)f - sib = 0
#  y - skf =0 # Where k == n_iterations
#
#  sif = Relu(sib)
def define_positive_sum_network(xlim, ylim, n_iterations):
    '''
    Defines the positive_sum network in a marabou way
        s_i = ReLu(1 * x_i + 1 * s_i-1)
        y = s_k (where k == n_iterations)
    :param xlim: how to limit the input to the network
    :param ylim: how to limit the output of the network
    :param n_iterations: number of inputs / times the rnn cell will be executed
    :return: query to marabou that defines the positive_sum rnn network (without recurent)
    '''
    positive_sum_rnn_query = MarabouCore.InputQuery()
    positive_sum_rnn_query.setNumberOfVariables(1)  # x

    # x
    positive_sum_rnn_query.setLowerBound(0, xlim[0])
    positive_sum_rnn_query.setUpperBound(0, xlim[1])

    rnn_start_idx = 1  # i
    rnn_idx = add_rnn_cell(positive_sum_rnn_query, [(0, 1)], 1, n_iterations, print_debug=1)  # rnn_idx == s_i f
    s_i_1_f_idx = rnn_idx - 2
    y_idx = rnn_idx + 1

    positive_sum_rnn_query.setNumberOfVariables(y_idx + 1)

    # y
    positive_sum_rnn_query.setLowerBound(y_idx, -large)
    positive_sum_rnn_query.setUpperBound(y_idx, large)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_idx)
    output_equation.addAddend(-1, rnn_idx)
    output_equation.setScalar(0)
    # output_equation.dump()
    positive_sum_rnn_query.addEquation(output_equation)

    # s_i f <= i + 1 <--> i + 1 - s_i f >= 0
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    invariant_equation.addAddend(1, rnn_start_idx)  # i
    invariant_equation.addAddend(-1, rnn_idx)  # s_i f
    invariant_equation.setScalar(-1)

    # y <= ylim
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, y_idx)
    property_eq.setScalar(ylim[1])

    return positive_sum_rnn_query, [rnn_start_idx], invariant_equation, [property_eq]

def define_positive_sum_linear_input_constraint(x_addition, x_time_multiply, ylim, n_iterations):
    '''
    Defines the positive_sum network in a marabou way
        s_i = ReLu(1 * x_i + 1 * s_i-1)
        y = s_k (where k == n_iterations)
    We bound x to be c*i+b (where i is the time stamp)
    :param x_addition: tuple with two entries (min, max), of the b value in x bounds
    :param x_time_multiply: tuple with two entries (min, max), of the c value in x bounds
    :param ylim: how to limit the output of the network
    :param n_iterations: number of inputs / times the rnn cell will be executed
    :return: query to marabou that defines the positive_sum rnn network (without recurent)
    '''
    positive_sum_rnn_query = MarabouCore.InputQuery()
    positive_sum_rnn_query.setNumberOfVariables(1)  # x

    rnn_start_idx = 1  # i
    rnn_idx = add_rnn_cell(positive_sum_rnn_query, [(0, 1)], 1, n_iterations, print_debug=1)  # rnn_idx == s_i f
    s_i_1_f_idx = rnn_idx - 2
    y_idx = rnn_idx + 1


    # x bounds
    x_bound_min_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    x_bound_min_eq.addAddend(1, 0) # x
    x_bound_min_eq.addAddend(-x_time_multiply[0], rnn_start_idx) # c * i
    x_bound_min_eq.setScalar(x_addition[0])
    positive_sum_rnn_query.addEquation(x_bound_min_eq)

    x_bound_max_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    x_bound_max_eq.addAddend(1, 0)  # x
    x_bound_max_eq.addAddend(-x_time_multiply[1], rnn_start_idx)  # c * i
    x_bound_max_eq.setScalar(x_addition[1])
    positive_sum_rnn_query.addEquation(x_bound_max_eq)

    def relu(x):
        return max(x, 0)

    positive_sum_rnn_query.setNumberOfVariables(y_idx + 1)

    # y
    positive_sum_rnn_query.setLowerBound(y_idx, -large)
    positive_sum_rnn_query.setUpperBound(y_idx, large)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_idx)
    output_equation.addAddend(-1, rnn_idx)
    output_equation.setScalar(0)
    # output_equation.dump()
    positive_sum_rnn_query.addEquation(output_equation)

    # y <= ylim
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, y_idx)
    property_eq.setScalar(ylim[1])

    min_y_0 = relu(relu(x_addition[0] + x_time_multiply[0] * 1) * 1)
    max_y_0 = relu(relu(x_addition[1] + x_time_multiply[1] * 1) * 1)

    initial_values = [max_y_0]

    return positive_sum_rnn_query, [rnn_start_idx], None, [property_eq], initial_values

def define_positive_sum_network_no_invariant(xlim, ylim, n_iterations):
    '''
    Defines the positive_sum network in a marabou way
        s_i = ReLu(1 * x_i + 1 * s_i-1)
        y = s_k (where k == n_iterations)
    :param xlim: how to limit the input to the network
    :param ylim: how to limit the output of the network
    :param n_iterations: number of inputs / times the rnn cell will be executed
    :return: query to marabou that defines the positive_sum rnn network (without recurent)
    '''
    positive_sum_rnn_query = MarabouCore.InputQuery()
    positive_sum_rnn_query.setNumberOfVariables(1)  # x

    # x
    positive_sum_rnn_query.setLowerBound(0, xlim[0])
    positive_sum_rnn_query.setUpperBound(0, xlim[1])

    rnn_start_idx = 1  # i
    rnn_idx = add_rnn_cell(positive_sum_rnn_query, [(0, 1)], 1, n_iterations, print_debug=1)  # rnn_idx == s_i f
    s_i_1_f_idx = rnn_idx - 2
    y_idx = rnn_idx + 1

    def relu(x):
        return max(x, 0)

    positive_sum_rnn_query.setNumberOfVariables(y_idx + 1)

    # y
    positive_sum_rnn_query.setLowerBound(y_idx, -large)
    positive_sum_rnn_query.setUpperBound(y_idx, large)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_idx)
    output_equation.addAddend(-1, rnn_idx)
    output_equation.setScalar(0)
    # output_equation.dump()
    positive_sum_rnn_query.addEquation(output_equation)

    # y <= ylim
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, y_idx)
    property_eq.setScalar(ylim[1])

    min_y = relu(relu(xlim[0] * 1) * 1)
    max_y = relu(relu(xlim[1] * 1) * 1)

    initial_values = [max_y]

    return positive_sum_rnn_query, [rnn_start_idx], None, [property_eq], initial_values


def define_last_network(xlim, ylim, n_iterations):
    '''
    Function that define "last_network" which is an RNN network that outputs the last input parameter
    :param xlim: how to limit the input to the network
    :param ylim: how to limit the output of the network
    :param n_iterations: number of inputs / times the rnn cell will be executed
    :return: (network, [rnn output indices], invariant equation, output equation
    '''
    query = MarabouCore.InputQuery()
    query.setNumberOfVariables(1)

    # x
    query.setLowerBound(0, xlim[0])
    query.setUpperBound(0, xlim[1])

    # rnn, the s_i = 0 * s_i-1 + x * 1
    rnn_idx = add_rnn_cell(query, [(0, 1)], 0, n_iterations)
    y_idx = rnn_idx + 1

    query.setNumberOfVariables(y_idx + 1)
    # y
    query.setLowerBound(y_idx, -large)
    query.setUpperBound(y_idx, large)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_idx)
    output_equation.addAddend(-1, rnn_idx)
    output_equation.setScalar(0)
    # output_equation.dump()
    query.addEquation(output_equation)

    # s_i-1 f <= xlim[1]
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, rnn_idx - 2)  # s_i-1 f
    invariant_equation.setScalar(xlim[1])

    # y <= ylim
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, y_idx)
    property_eq.setScalar(ylim[1])

    return query, [rnn_idx], invariant_equation, [property_eq]


def define_two_sum_network(xlim, ylim, n_ierations):
    '''
    The network gets a series of numbers and outputs two neurons, one sums the positive numbers and the other
    the negative
    The property we will
    :param xlim: how to limit the input to the network
    :param ylim: how to limit the output of the network
    :param n_iterations: number of inputs / times the rnn cell will be executed
    :return: query to marabou that defines the positive_sum rnn network (without recurent)
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(1)  # x

    # x
    network.setLowerBound(0, xlim[0])
    network.setUpperBound(0, xlim[1])

    rnn_start_idx = 1  # i
    rnn_idx = add_rnn_cell(network, [(0, 1)], 1, n_ierations)  # rnn_idx == s_i f
    y_idx = rnn_idx + 1

    network.setNumberOfVariables(y_idx + 1)

    # y
    network.setLowerBound(y_idx, -large)
    network.setUpperBound(y_idx, large)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_idx)
    output_equation.addAddend(-1, rnn_idx)
    output_equation.setScalar(0)
    # output_equation.dump()
    network.addEquation(output_equation)

    # s_i f <= i <--> i - s_i f >= 0
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    invariant_equation.addAddend(1, rnn_start_idx)  # i
    invariant_equation.addAddend(-1, rnn_idx)  # s_i f
    invariant_equation.setScalar(0)

    # y <= ylim
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, y_idx)
    property_eq.setScalar(ylim[1])

    return network, [rnn_start_idx], invariant_equation, [property_eq]


def define_concatenate_rnn_invariant_not_holding(xlim, ylim, n_iterations):
    '''
    defining a network with two rnn's one after the other, so the input to the second rnn is the output of the first
    Here the invariant of the second rnn does not hold.
    :param xlim:
    :param ylim:
    :param n_iterations:
    :return:
    '''
    positive_sum_rnn_query = MarabouCore.InputQuery()
    positive_sum_rnn_query.setNumberOfVariables(1)  # x

    # x
    positive_sum_rnn_query.setLowerBound(0, xlim[0])
    positive_sum_rnn_query.setUpperBound(0, xlim[1])

    rnn_1_start_idx = 1  # i
    rnn_1_idx = add_rnn_cell(positive_sum_rnn_query, [(0, 1)], 1, n_iterations, print_debug=1)  # rnn_idx == s_i f
    rnn_2_start_idx = rnn_1_idx + 1  # i
    rnn_2_idx = add_rnn_cell(positive_sum_rnn_query, [(rnn_1_idx, 1)], 1, n_iterations,
                             print_debug=1)  # rnn_idx == s_i f
    y_idx = rnn_2_idx + 1

    positive_sum_rnn_query.setNumberOfVariables(y_idx + 1)

    # y
    positive_sum_rnn_query.setLowerBound(y_idx, -large)
    positive_sum_rnn_query.setUpperBound(y_idx, large)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_idx)
    output_equation.addAddend(-1, rnn_2_idx)
    output_equation.setScalar(0)
    # output_equation.dump()
    positive_sum_rnn_query.addEquation(output_equation)

    # s_i_f >= i + 1 <--> -1 >= - s_i_f + i
    invariant_1_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    invariant_1_equation.addAddend(1, rnn_1_start_idx)  # i
    invariant_1_equation.addAddend(-1, rnn_1_idx)  # s_i f
    invariant_1_equation.setScalar(-1)

    # z_i f >= n_iterations * (i + 1) <--> -n >= n * i - z_i_f
    invariant_2_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    invariant_2_equation.addAddend(n_iterations + 1, rnn_2_start_idx)  # i
    invariant_2_equation.addAddend(-1, rnn_2_idx)  # z_i f
    invariant_2_equation.setScalar(-1)

    # y <= ylim
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, y_idx)
    property_eq.setScalar(ylim[1])

    return positive_sum_rnn_query, [rnn_1_start_idx, rnn_2_start_idx], [invariant_1_equation, invariant_2_equation], [
        property_eq]


def define_concatenate_rnn(xlim, ylim, n_iterations):
    '''
        xlim[0] <= x_0 <= xlim[1]
        s_i = 1 * x_0 + 1 * s_i-1
        z_i = 1 * s_i + 1 * z_i-1
        y = z_i
    '''
    query = MarabouCore.InputQuery()
    query.setNumberOfVariables(1)  # x

    # x
    query.setLowerBound(0, xlim[0])
    query.setUpperBound(0, xlim[1])

    rnn_1_start_idx = 1  # i
    rnn_1_idx = add_rnn_cell(query, [(0, 1)], 1, n_iterations, print_debug=1)  # rnn_idx == s_i f
    rnn_2_start_idx = rnn_1_idx + 1  # i
    rnn_2_idx = add_rnn_cell(query, [(rnn_1_idx, 1)], 1, n_iterations, print_debug=1)  # rnn_idx == s_i f
    y_idx = rnn_2_idx + 1
    def relu(x):
        return max(0, x)
    min_s1 = relu(xlim[0] * 1)
    max_s1 = relu(xlim[1] * 1)
    min_z1 = relu(min_s1 * 1)
    max_z1 = relu(max_s1 * 1)

    query.setNumberOfVariables(y_idx + 1)

    # y
    query.setLowerBound(y_idx, -large)
    query.setUpperBound(y_idx, large)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_idx)
    output_equation.addAddend(-1, rnn_2_idx)
    output_equation.setScalar(0)
    # output_equation.dump()
    query.addEquation(output_equation)

    # s_i_f >= i + 1 <--> -1 >= - s_i_f + i
    invariant_1_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    invariant_1_equation.addAddend(1, rnn_1_start_idx)  # i
    invariant_1_equation.addAddend(-1, rnn_1_idx)  # s_i f
    invariant_1_equation.setScalar(-1)

    # z_i f >= n_iterations * (i + 1) <--> -n >= n * i - z_i_f
    invariant_2_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    invariant_2_equation.addAddend(n_iterations + 1, rnn_2_start_idx)  # i
    invariant_2_equation.addAddend(-1, rnn_2_idx)  # z_i f
    invariant_2_equation.setScalar(-1)

    # y <= ylim
    property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    property_eq.addAddend(1, y_idx)
    property_eq.setScalar(ylim[1])

    return query, [rnn_1_start_idx, rnn_2_start_idx], [invariant_1_equation, invariant_2_equation], [
        property_eq], [max_s1, max_z1]


def test_negate_equation_GE():
    # x - y >= 0
    eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    eq.addAddend(1, 1)
    eq.addAddend(-1, 0)  # i
    eq.setScalar(0)

    # x - y <= -epsilon
    not_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    not_eq.addAddend(1, 1)  # s_i b
    not_eq.addAddend(-1, 0)  # i
    not_eq.setScalar(-SMALL)
    actual_not_eq = negate_equation(eq)

    assert actual_not_eq.equivalent(not_eq)
    assert not eq.equivalent(not_eq)


def test_negate_equation_LE():
    # x + y <= 1
    eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    eq.addAddend(1, 0)
    eq.addAddend(1, 1)
    eq.setScalar(1)

    # x + y >= 1 + epsilon
    not_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    not_eq.addAddend(1, 0)
    not_eq.addAddend(1, 1)
    not_eq.setScalar(1 + SMALL)
    actual_not_eq = negate_equation(eq)

    assert actual_not_eq.equivalent(not_eq)
    assert not eq.equivalent(not_eq)


def test_negative_sum_invariant_not_hold():
    num_iterations = 500
    invariant_xlim = (-1.1, -0.9)
    y_lim = (0, num_iterations)

    assert not prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_negative_sum_network)


def test_negative_sum_positive():
    num_iterations = 500
    invariant_xlim = (-1, 1)
    y_lim = (0, num_iterations + 1)

    assert prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_negative_sum_network)


def test_positive_sum_base_not_hold():
    num_iterations = 500
    invariant_xlim = (1, 1.1)
    y_lim = (0, num_iterations)

    assert not prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_positive_sum_network)


def test_positive_sum_property_not_hold():
    num_iterations = 500
    invariant_xlim = (0, 1)
    y_lim = (0, num_iterations // 2)

    assert not prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_positive_sum_network)


def test_positive_sum_positive():
    num_iterations = 500
    invariant_xlim = (-1, 1)
    y_lim = (0, num_iterations + 1)

    assert prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_positive_sum_network)


def test_auto_positive_sum_linear_input_constraint_positive():
    num_iterations = 4
    x_addition = (0, 0)
    x_multiply = (0.5, 1)
    max_y = 0
    for i in range(num_iterations):
        max_y += x_multiply[1] * i + x_addition[1]
    ylim = (0, 16)

    positive_sum_rnn_query, rnn_start_idx, invariant_equation, property_eqs, initial_values = define_positive_sum_linear_input_constraint(
        x_addition, x_multiply, ylim, num_iterations)

    assert find_invariant_marabou(positive_sum_rnn_query, rnn_start_idx, [MarabouCore.Equation.LE], initial_values,
                           num_iterations, property_eqs)

def test_auto_positive_sum_negative():
    num_iterations = 500
    xlim = (-1, 1.1)
    ylim = (0, num_iterations + 1)

    positive_sum_rnn_query, rnn_start_idx, invariant_equation, property_eqs, initial_values = define_positive_sum_network_no_invariant(
        xlim, ylim, num_iterations)

    assert not find_invariant_marabou(positive_sum_rnn_query, rnn_start_idx, [MarabouCore.Equation.LE], initial_values,
                           num_iterations, property_eqs)


def test_auto_positive_sum_positive():
    num_iterations = 500
    xlim = (-1, 1)
    ylim = (0, num_iterations + 1)

    positive_sum_rnn_query, rnn_start_idx, invariant_equation, property_eqs, initial_values = define_positive_sum_network_no_invariant(
        xlim, ylim, num_iterations)

    assert find_invariant_marabou(positive_sum_rnn_query, rnn_start_idx, [MarabouCore.Equation.LE], initial_values,
                           num_iterations, property_eqs)


# def test_last_network_negative():
#     num_iterations = 500
#     invariant_xlim = (-1, 2)
#     y_lim = (-1, 0)
#     assert not prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_last_network)
#
#
# def test_last_network_positive():
#     '''
#     create wanted property and invariant that holds
#     :return:
#     '''
#     num_iterations = 500
#     invariant_xlim = (-1, 1)
#     y_lim = invariant_xlim
#     assert prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_last_network)

# @pytest.mark.slow

# def test_zero_network_positive():
#     num_iterations = 500
#     invariant_xlim = (-1, 1)
#     y_lim = 10 ** -2
#     assert prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_zero_network)


def test_concatenate_rnn_cells_positive():
    # for i in range(1, 10):
    i = 5
    num_iterations = i  # we use i = 0 so it's 6 iterations
    invariant_xlim = (0, 1)
    y_lim = (0, (num_iterations + 1) ** 2)
    assert prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_concatenate_rnn)
    print("prove that for {} R_2 is <= {}".format(num_iterations, (num_iterations + 1) ** 2))


def test_concatenate_rnn_cells_positive_output_fail():
    num_iterations = 5
    invariant_xlim = (-1, 1)
    y_lim = (0, 10000)
    assert not prove_using_invariant(invariant_xlim, y_lim, num_iterations,
                                     define_concatenate_rnn_invariant_not_holding)


def test_concatenate_rnn_cells_positive_output_fail():
    num_iterations = 5
    invariant_xlim = (-1, 1)
    y_lim = (0, 15)
    assert not prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_concatenate_rnn)


def test_auto_positive_sum_positive():
    '''
    0: x0
    1: i
    2: s_i-1
    3: s_i_b
    4: s_i_f
    5: i
    6: z_i-1
    7: z_i_b
    8: z_i_f
    9: out
    :return:
    '''
    i = 5
    num_iterations = i  # we use i = 0 so it's 6 iterations
    invariant_xlim = (0, 1)
    y_lim = (0, (num_iterations + 1) ** 2)
    query, rnn_start_idx, _, property_eqs, initial_values = \
        define_concatenate_rnn(invariant_xlim, y_lim, num_iterations)
    rnn_dependent = [[1], None]
    assert find_invariant_marabou(query, rnn_start_idx, [MarabouCore.Equation.LE, MarabouCore.Equation.LE],
                                      initial_values,
                                      num_iterations, property_eqs, rnn_dependent=rnn_dependent)


def test_auto_positive_sum_negative():
    '''
    0: x0
    1: i
    2: s_i-1
    3: s_i_b
    4: s_i_f
    5: i
    6: z_i-1
    7: z_i_b
    8: z_i_f
    9: out
    :return:
    '''
    num_iterations = 5
    invariant_xlim = (-1, 1)
    y_lim = (0, 15)
    query, rnn_start_idx, _, property_eqs, initial_values = \
        define_concatenate_rnn(invariant_xlim, y_lim, num_iterations)
    rnn_dependent = [[1], None]
    assert not find_invariant_marabou(query, rnn_start_idx, [MarabouCore.Equation.LE, MarabouCore.Equation.LE],
                                      initial_values,
                                      num_iterations, property_eqs, rnn_dependent=rnn_dependent)
