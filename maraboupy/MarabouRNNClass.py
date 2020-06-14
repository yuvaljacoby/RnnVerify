import numpy as np

from maraboupy import MarabouCore
from maraboupy.MarabouRNN import marabou_solve_negate_eq

large = 500000.0
small = 10 ** -2
TOLERANCE_VALUE = 0.01
ALPHA_IMPROVE_EACH_ITERATION = 10



def add_rnn_multidim_cells(query, input_idx, input_weights, hidden_weights, bias, num_iterations, print_debug=False):
    '''
    Create n rnn cells, where n is hidden_weights.shape[0] == hidden_weights.shape[1] ==  len(bias)
    The added parameters are (same order): i, s_i-1 f, s_i b, s_i f for each of the n added cells (i.e. adding 4*n variables)
    :param query: the network so far (will add to this)
    :param input_idx: list of input id's, length m
    :param input_weights: matrix of input weights, size m x n
    :param hidden_weights: matrix of weights
    :param bias: vector of biases to add to each equation, length should be n, if None use 0 bias
    :param num_iterations: Number of iterations
    :return: list of output cells, the length will be the same n
    '''
    assert type(hidden_weights) == np.ndarray
    assert len(hidden_weights.shape) == 2
    assert hidden_weights.shape[0] == hidden_weights.shape[1]
    assert len(input_idx) == input_weights.shape[1]
    assert hidden_weights.shape[0] == input_weights.shape[0]

    n = hidden_weights.shape[0]
    if bias is None:
        bias = [0] * n
    else:
        assert len(bias) == n

    last_idx = query.getNumberOfVariables()
    prev_iteration_idxs = [i + 1 for i in range(last_idx, last_idx + (4 * n), 4)]
    output_idxs = [i + 3 for i in range(last_idx, last_idx + (4 * n), 4)]
    query.setNumberOfVariables(last_idx + (4 * n))  # i, s_i-1 f, s_i b, s_i f

    added_cells = []
    cell_idx = last_idx
    for i in range(n):
        added_cells.append(MarabouRnn(query, num_iterations, input_idx, input_weights[i, :], prev_iteration_idxs,
                                      hidden_weights[i, :], cell_idx))

        cell_idx += 4


class MarabouRnn:
    def _add_variables(self, query, n):
        cell_idx = query.getNumberOfVariables()

        query.setLowerBound(cell_idx, 0)
        query.setUpperBound(cell_idx, n)
        self.iterator_idx = cell_idx

        # s_i-1 f
        query.setLowerBound(cell_idx + 1, 0)
        query.setUpperBound(cell_idx + 1, large)
        self.prev_iteration_idx = cell_idx + 1

        # s_i b
        query.setLowerBound(cell_idx + 2, -large)
        query.setUpperBound(cell_idx + 2, large)

        # s_i f
        query.setLowerBound(cell_idx + 3, 0)
        query.setUpperBound(cell_idx + 3, large)
        self.output_idx = cell_idx

    def _add_update_equation(self, input_idx, input_weights, hidden_idx, hidden_weights, bias):
        '''
        Creating an update equation for this rnn cell

        :param input_idx: list size n with indices of the input variables
        :param input_weights: list size n, weight for each of the indices from input_idx (same order)
        :param hidden_idx: list size m with indices of the hidden variables
        :param hidden_weights: list size m, weight for each of the indices from hidden_idx (same order)
        :param bias: the bias to add to the update equation
        :return:
        '''
        self.update_eq = MarabouCore.Equation()
        for j in range(len(input_weights)):
            self.update_eq.addAddend(input_weights[j], input_idx[j])

        for j, w in enumerate(hidden_weights):
            self.update_eq.addAddend(w, hidden_idx[j])

        self.update_eq.addAddend(-1, self.iterator_idx + 2)
        self.update_eq.setScalar(-bias)

    def __init__(self, query: MarabouCore.InputQuery, invariant_type, num_iterations, input_idx, input_weights, hidden_idx,
                 hidden_weights, bias, start_idx=None):
        '''
        Create an RNN cell object for the given query for number of iterations
        Not adding to the query equations anything (use addQuery / removeQuery) to do so
        Adding 4 variables to it
        :param query: A marabou InputQuery object that this rnn will work with
        :param num_iterations: maximum number of iterations
        :param input_idx: list size n with indices of the input variables
        :param input_weights: list size n, weight for each of the indices from input_idx (same order)
        :param hidden_idx: list size m with indices of the hidden variables
        :param hidden_weights: list size m, weight for each of the indices from hidden_idx (same order)
        :param bias: the bias to add to the update equation
        :param start_idx: First index to use, to get the last use self.get_output_idx, if None using the last index in the query
        '''
        self._add_variables(query, num_iterations, start_idx)
        self._add_update_equation(input_idx, input_weights, hidden_idx, hidden_weights, bias)

        assert invariant_type == MarabouCore.Equation.GE or invariant_type == MarabouCore.Equation.LE
        self.invariant_type = invariant_type

        query.addEquation(self.update_eq)

    def get_max_alpha(self):
        return self.max_alpha if self.max_alpha else large

    def set_max_alpha(self, new_max_alpha):
        if new_max_alpha < self.get_max_alpha():
            self.max_alpha = new_max_alpha
        else:
            raise Exception()

    def set_min_alpha(self, new_min_alpha):
        if new_min_alpha > self.get_min_alpha():
            self.max_alpha = new_min_alpha
        else:
            raise Exception()

    def get_min_alpha(self):
        return self.min_alpha if self.min_alpha else -large

    def get_last_alpha(self):
        if self.alpha:
            return self.alpha
        else:
            if self.invariant_type == MarabouCore.Equation.GE:
                return self.get_min_alpha()
            else:
                return self.get_max_alpha()

    def get_last_invariant(self):
        if self.eq_invariant:
            return self.eq_invariant
        else:
            return None

    def get_invariant_base_eq(self):
        pass

    def get_invariant_step_eq(self):
        pass

    # def get_iterator_idx(self):
    #     return self.iterator_idx
    #
    # def get_output_idx(self):
    #     return self.output_idx
