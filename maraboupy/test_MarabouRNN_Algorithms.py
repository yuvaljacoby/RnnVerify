from rnn_algorithms.IterateAlphasSGD import IterateAlphasSGD
from rnn_algorithms.SMTBaseSearch import SmtAlphaSearch
from maraboupy import MarabouCore
from maraboupy.MarabouRNN import *
from RNN.MarabouRNNMultiDim import prove_multidim_property


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

    initial_values = [[min_y], [max_y]]

    return positive_sum_rnn_query, [rnn_start_idx], None, [negate_equation(property_eq)], initial_values


def test_auto_positive_sum_positive_iterateSGD():
    num_iterations = 10
    xlim = (-1, 1)
    ylim = (0, num_iterations + 1.1)

    network, rnn_start_idxs, invariant_equation, property_eqs, initial_values = define_positive_sum_network_no_invariant(
        xlim, ylim, num_iterations)
    rnn_output_idxs = [i + 3 for i in rnn_start_idxs]

    algorithm = IterateAlphasSGD(initial_values, rnn_start_idxs, rnn_output_idxs)
    assert prove_multidim_property(network, rnn_start_idxs, rnn_output_idxs, property_eqs, algorithm)



def auto_positive_sum_positive_SMTbase():
    num_iterations = 10
    xlim = (-1, 1)
    ylim = (0, num_iterations + 1.1)

    network, rnn_start_idxs, invariant_equation, property_eqs, initial_values = define_positive_sum_network_no_invariant(
        xlim, ylim, num_iterations)
    rnn_output_idxs = [i + 3 for i in rnn_start_idxs]

    algorithm = SmtAlphaSearch(initial_values, rnn_start_idxs, rnn_output_idxs, np.array([[1]]), np.array([[1]]),
                               [0], [xlim[0]], [xlim[1]], num_iterations)
    assert prove_multidim_property(network, rnn_start_idxs, rnn_output_idxs, property_eqs, algorithm)


if __name__ == "__main__":
    test_auto_positive_sum_positive_iterateSGD()
    # auto_positive_sum_positive_SMTbase()