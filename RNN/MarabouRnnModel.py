import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from maraboupy import MarabouCore
from datetime import datetime

MARABOU_TIMEOUT = 120

SMALL = 10 ** -2
LARGE = 500


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
    assert len(input_idx) == input_weights.shape[1], "{}, {}".format(len(input_idx), input_weights.shape[1])
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

    cell_idx = last_idx
    for i in range(n):
        # i
        query.setLowerBound(cell_idx, 0)
        # we bound the memory unit, and num_iterations is for the output which is doing one more calculation
        query.setUpperBound(cell_idx, num_iterations - 1)

        # s_i-1 f
        query.setLowerBound(cell_idx + 1, 0)
        query.setUpperBound(cell_idx + 1, LARGE)

        # s_i b
        query.setLowerBound(cell_idx + 2, -LARGE)
        query.setUpperBound(cell_idx + 2, LARGE)

        # s_i f
        query.setLowerBound(cell_idx + 3, 0)
        query.setUpperBound(cell_idx + 3, LARGE)

        # s_i f = ReLu(s_i b)
        MarabouCore.addReluConstraint(query, cell_idx + 2, cell_idx + 3)

        # s_i-1 f >= i * \sum (x_j_min * w_j)
        # prev_min_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
        # prev_min_eq.addAddend(1, last_idx + 1)
        # prev_min_eq.addAddend(1, last_idx + 1)

        # s_i b = x_j * w_j for all j connected + s_i-1 f * hidden_weight
        update_eq = MarabouCore.Equation()
        for j in range(len(input_weights[i, :])):
            w = input_weights[i, j]  # round(input_weights[i, j], 2)
            update_eq.addAddend(w, input_idx[j])

        for j, w in enumerate(hidden_weights[i, :]):
            # w = round(w, 2)
            update_eq.addAddend(w, prev_iteration_idxs[j])

        update_eq.addAddend(-1, cell_idx + 2)
        update_eq.setScalar(-bias[i])
        # if print_debug:
        #     update_eq.dump()
        query.addEquation(update_eq)
        cell_idx += 4

    return output_idxs


def relu(x):
    return max(x, 0)


class RnnMarabouModel():
    def __init__(self, h5_file_path, n_iterations=10):
        self.network = MarabouCore.InputQuery()
        self.model = tf.keras.models.load_model(h5_file_path)
        # TODO: If the input is 2d wouldn't work
        n_input_nodes = self.model.input_shape[-1]
        prev_layer_idx = list(range(0, n_input_nodes))
        self.input_idx = prev_layer_idx
        self.n_iterations = n_iterations
        self.rnn_out_idx = []

        # Each cell in the list is a triple (in_w, hidden_w, bias), the cells are sorted by layer from input to output
        self.rnn_weights = []

        # save spot for the input nodes
        self.network.setNumberOfVariables(n_input_nodes)
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.SimpleRNN):
                prev_layer_idx = self.add_rnn_simple_layer(layer, prev_layer_idx)
                self.rnn_out_idx.append(prev_layer_idx)
            elif type(layer) == tf.keras.layers.Dense:
                prev_layer_idx = self.add_dense_layer(layer, prev_layer_idx)
            else:
                #
                raise NotImplementedError("{} layer is not supported".format(type(layer)))

        # Save the last layer output indcies
        self.output_idx = list(range(*prev_layer_idx))
        self._rnn_loop_idx = []
        self._rnn_prev_iteration_idx = []
        for layer_out_idx in self.rnn_out_idx:
            self._rnn_loop_idx.append([i - 3 for i in layer_out_idx])
            self._rnn_prev_iteration_idx.append([i - 2 for i in layer_out_idx])

        self.num_rnn_layers = len(self.rnn_out_idx)

    def __del__(self):
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass

    def get_start_end_idxs(self, rnn_layer=0):
        # rnn_start_idxs = []
        # rnn_start_idxs = [i - 3 for i in self.rnn_out_idx]
        if rnn_layer is None:
            return self._rnn_loop_idx, self.rnn_out_idx
        return self._rnn_loop_idx[rnn_layer], self.rnn_out_idx[rnn_layer]

    def _get_value_by_layer(self, layer, input_tensor):
        intermediate_layer_model = Model(inputs=self.model.input, outputs=layer.output)
        intermediate_output = intermediate_layer_model.predict(input_tensor)

        # assert len(input_tensor.shape) == len(self.model.input_shape)
        # assert sum([input_tensor.shape[i] == self.model.input_shape[i] or self.model.input_shape[i] is None
        #         for i in range(len(input_tensor.shape))]) == len(input_tensor.shape)
        # out_ten = layer.output
        # with tf.Session() as sess:
        #     tf.global_variables_initializer().run()
        #     output = sess.run(out_ten, {self.model.input: input_tensor})

        return np.squeeze(intermediate_output)

    def _get_rnn_value_one_iteration(self, in_tensor):
        outputs = []
        for l in self.model.layers:
            if type(l) == tf.keras.layers.SimpleRNN:
                outputs.append(self._get_value_by_layer(l, in_tensor))
        return outputs

    def find_max_by_weights(self, xlim, layer):
        total = 0
        for i, w in enumerate(layer.get_weights()[0]):
            if w < 0:
                total += xlim[i][0] * w
            else:
                total += xlim[i][1] * w
        return total

    def get_rnn_max_value_one_iteration(self, xlim):
        xlim_max = [x[1] for x in xlim]
        max_in_tensor = np.array(xlim_max)[None, None, :]  # add two dimensions
        return self._get_rnn_value_one_iteration(max_in_tensor)

    def get_weights(self):
        '''
        :return: list of tuples each is (w_in, w_h, b) of a rnn layer from input to output
        '''
        all_weights = []
        for i, layer in enumerate(self.model.layers):
            if type(layer) == tf.keras.layers.SimpleRNN:
                w_in, w_h, b = layer.get_weights()
                all_weights.append((w_in, w_h, b))
        return all_weights

    # def get_rnn_min_max_value_one_iteration(self, xlim, layer_idx=0):
    #     for i, layer in enumerate(self.model.layers[:layer_idx + 1]):
    #         initial_values = []
    #         if type(layer) == tf.keras.layers.SimpleRNN:
    #             if i == 0:
    #                 # There is no non linarity and the constraints are simple just take upper bound if weight is positive
    #                 #  and lower otherwise
    #
    #                 # It's only one iteration so the hidden weights  is zeroed
    #                 in_w, _, b = layer.get_weights()
    #                 for i, rnn_dim_weights in enumerate(in_w.T):
    #                     max_val = 0
    #                     min_val = 0
    #                     for j, w in enumerate(rnn_dim_weights):
    #                         w = round(w, 6)
    #                         v1 = w * xlim[j][1]
    #                         v2 = w * xlim[j][0]
    #                         if v1 > v2:
    #                             max_val += v1
    #                             min_val += v2
    #                         else:
    #                             max_val += v2
    #                             min_val += v1
    #                     min_val += b[i]
    #                     max_val += b[i]
    #                     # TODO: +- SMALL is not ideal here (SMALL = 10**-2) but otherwise there are rounding problems
    #                     # min_val = relu(min_val) - 2 * SMALL if relu(min_val) > 0 else 0
    #
    #                     initial_values.append((relu(min_val), relu(max_val)))
    #                 # There are rounding problems between this calculation and marabou, query marabou to make sure it's OK
    #                 self.query_marabou_to_improve_values(initial_values)
    #             else:
    #                 # Need to query gurobi here...
    #                 raise NotImplementedError()
    #         # print('initial_values:', initial_values)
    #         # return initial_values
    #         rnn_max_values = [val[1] for val in initial_values]
    #         rnn_min_values = [val[0] for val in initial_values]
    #         # DEBUG
    #         assert sum([rnn_max_values[i] >= rnn_min_values[i] for i in range(len(rnn_max_values))]) == len(rnn_max_values)
    #         return (rnn_min_values, rnn_max_values)

    def get_rnn_min_max_value_one_iteration(self, prev_layer_alpha, layer_idx=0, prev_layer_beta=None):
        '''
        calculate the first iteration value of recurrent layer layer_idx using the bounds from the previous layer
        the previous layers bound is alpha*t + beta
        :param prev_layer_alpha:
        :param prev_layer_scalar:
        :param layer_idx:
        :return: (list of min values, list of max values)
        '''
        layer = self.model.layers[layer_idx]
        initial_values = []
        # There is no non linarity and the constraints are simple just take upper bound if weight is positive
        #  and lower otherwise

        # It's only one iteration so the hidden weights  is zeroed
        in_w, _, b = layer.get_weights()
        if prev_layer_beta is None:
            prev_layer_beta = ([0] * len(prev_layer_alpha), [0] * len(prev_layer_alpha))

        for i, rnn_dim_weights in enumerate(in_w.T):
            max_val = 0
            min_val = 0
            for j, w in enumerate(rnn_dim_weights):
                # w = round(w, 6)
                v_max_bound = w * (prev_layer_alpha[j][1] + prev_layer_beta[1][j])
                v_min_bound = w * (prev_layer_alpha[j][0] + prev_layer_beta[0][j])
                if layer_idx > 0 and w < 0:
                    # The invariant (prev_layer_bounds[j] is as a function of time, and here we are intrested in the first iteration where i=0
                    min_val += v_max_bound
                    max_val += prev_layer_beta[0][j]
                    continue
                if v_max_bound > v_min_bound:
                    max_val += v_max_bound
                    min_val += v_min_bound
                else:
                    max_val += v_min_bound
                    min_val += v_max_bound
            min_val += b[i]
            max_val += b[i]

            initial_values.append((relu(min_val), relu(max_val)))
            assert initial_values[-1][0] >= 0 and initial_values[-1][1] >= 0

            # There are rounding problems between this calculation and marabou, query marabou to make sure it's OK
            # Need to add bounds on the previous layer
            if layer_idx == 0:
                prev_layer_idx = self.input_idx
            else:
                prev_layer_idx = self.rnn_out_idx[layer_idx - 1]

            prev_layer_eqautions = []
            for i in range(len(prev_layer_alpha)):
                le_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
                le_eq.addAddend(1, prev_layer_idx[i])
                le_eq.setScalar(prev_layer_alpha[i][1])
                prev_layer_eqautions.append(le_eq)
                self.network.addEquation(le_eq)

                ge_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
                ge_eq.addAddend(1, prev_layer_idx[i])
                ge_eq.setScalar(prev_layer_alpha[i][0])
                prev_layer_eqautions.append(ge_eq)
                self.network.addEquation(ge_eq)

            self.query_marabou_to_improve_values(initial_values, layer_idx)

            for eq in prev_layer_eqautions:
                self.network.removeEquation(eq)

        # print('initial_values:', initial_values)
        # return initial_values
        rnn_max_values = [val[1] for val in initial_values]
        rnn_min_values = [val[0] for val in initial_values]
        # DEBUG
        assert sum([rnn_max_values[i] >= rnn_min_values[i] for i in range(len(rnn_max_values))]) == len(rnn_max_values)
        return (rnn_min_values, rnn_max_values)

    def query_marabou_to_improve_values(self, initial_values, layer_idx=0):
        def create_initial_run_equations(loop_indices, rnn_prev_iteration_idx):
            '''
            Zero the loop indcies and the rnn hidden values (the previous iteration output)
            :return: list of equations to add to marabou
            '''
            loop_equations = []
            for i in loop_indices:
                loop_eq = MarabouCore.Equation()
                loop_eq.addAddend(1, i)
                loop_eq.setScalar(0)
                loop_equations.append(loop_eq)

            # s_i-1 f == 0
            zero_rnn_hidden = []
            for idx in rnn_prev_iteration_idx:
                base_hypothesis = MarabouCore.Equation()
                base_hypothesis.addAddend(1, idx)
                base_hypothesis.setScalar(0)
                zero_rnn_hidden.append(base_hypothesis)
            return loop_equations + zero_rnn_hidden

        def improve_beta(eq, more_is_better):
            '''
            Run the equation on marabou until it is satisfied.
            If not satisfied taking the value from the index and using it as a s scalar
            using self.network to verify
            :param eq: Marabou equation of the form: +-1.000xINDEX >= SCALAR
            :param more_is_better: If true then adding epsilon on every fail, otherwise substracting
            :return: a scalar that satisfies the equation
            '''
            proved = False
            assert len(eq.getAddends()) == 1
            idx = eq.getAddends()[0].getVariable()
            beta = eq.getScalar()
            while not proved:
                eq.setScalar(beta)
                self.network.addEquation(eq)
                # print("{}: start improve query".format(str(datetime.now()).split(".")[0]), flush=True)
                vars1, stats1 = MarabouCore.solve(self.network, "", MARABOU_TIMEOUT, 0)
                # print("{}: finish improve  query".format(str(datetime.now()).split(".")[0]), flush=True)
                if stats1.hasTimedOut():
                    print("Marabou has timed out")
                    raise TimeoutError()
                # vars1, stats1 = MarabouCore.solve(self.network, "", 120, 0)
                if len(vars1) > 0:
                    proved = False
                    if more_is_better:
                        beta = vars1[idx] + SMALL
                    else:
                        beta = vars1[idx] - SMALL
                    # print("proof fail, trying with beta: {}".format(beta))
                else:
                    # print("UNSAT")
                    proved = True
                    # print("proof worked, with beta: {}".format(beta))
                    # self.network.dump()
                    # eq.dump()
                    # beta = beta
                self.network.removeEquation(eq)
            return beta

        all_loop_indcies = [i for layer in self._rnn_loop_idx for i in layer]
        initial_run_eq = create_initial_run_equations(all_loop_indcies, self._rnn_prev_iteration_idx[layer_idx])
        for init_eq in initial_run_eq:
            self.network.addEquation(init_eq)

        # not(R_i_f >= beta) <-> R_i_f <>= beta - epsilon
        for i in range(len(initial_values)):

            beta_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
            beta_eq.addAddend(1, self.rnn_out_idx[layer_idx][i])
            beta_eq.setScalar(initial_values[i][0] - SMALL)
            min_val = improve_beta(beta_eq, False)

            beta_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
            beta_eq.addAddend(1, self.rnn_out_idx[layer_idx][i])
            beta_eq.setScalar(initial_values[i][1] + SMALL)
            max_val = improve_beta(beta_eq, True)

            if min_val < 0:
                min_val = 0

            initial_values[i] = (min_val, max_val)

        for init_eq in initial_run_eq:
            self.network.removeEquation(init_eq)

    def get_rnn_min_max_value_one_iteration_marabou(self, xlim):
        xlim_min = [x[0] for x in xlim]
        min_in_tensor = np.array(xlim_min)[None, None, :]  # add two dimensions
        xlim_max = [x[1] for x in xlim]
        max_in_tensor = np.array(xlim_max)[None, None, :]  # add two dimensions
        # TODO: remove the assumption we have only one layer of RNN for initial_values (do we need to remove it?)
        max_values = self._get_rnn_value_one_iteration(max_in_tensor)[0]
        min_values = self._get_rnn_value_one_iteration(min_in_tensor)[0]
        # Not sure that the max_in tensor will yield the cell value to be LARGEr then the min_in_tensor
        initial_values = []
        for i in range(len(max_values)):
            if max_values[i] >= min_values[i]:
                initial_values.append((min_values[i], max_values[i]))
            else:
                initial_values.append((max_values[i], min_values[i]))

        # query_marabou_to_improve_values()
        return initial_values

    def add_rnn_simple_layer(self, layer, prev_layer_idx):
        '''
        Append to the marabou encoding (self.netowrk) a SimpleRNN layer
        :param layer: The layer object (using get_weights, expecting to get list length 3 of ndarrays)
        :param prev_layer_idx: Marabou indcies of the previous layer (to use to create the equations)
        :return: List of the added layer output variables
        '''
        if layer.activation == tf.keras.activations.relu:
            # Internal layer
            pass
        elif layer.activation == tf.keras.activations.softmax:
            # last layer
            raise NotImplementedError("activation {} is not supported".format(layer.activation.__name__))
        else:
            raise NotImplementedError("activation {} is not supported".format(layer.activation.__name__))

        rnn_input_weights, rnn_hidden_weights, rnn_bias_weights = layer.get_weights()
        self.rnn_weights.append((rnn_input_weights, rnn_hidden_weights, rnn_bias_weights))
        assert rnn_hidden_weights.shape[0] == rnn_input_weights.shape[1]
        assert rnn_hidden_weights.shape[0] == rnn_hidden_weights.shape[1]
        assert rnn_hidden_weights.shape[1] == rnn_bias_weights.shape[0]
        # first_idx = self.network.getNumberOfVariables()
        # self.network.setNumberOfVariables(first_idx + rnn_hidden_weights.shape[1])

        # TODO: Get number of iterations and pass it here instead of the 10
        output_idx = add_rnn_multidim_cells(self.network, prev_layer_idx, rnn_input_weights.T, rnn_hidden_weights,
                                            rnn_bias_weights, self.n_iterations)
        return output_idx

    def add_dense_layer(self, layer, prev_layer_idx):
        output_weights, output_bias_weights = layer.get_weights()
        assert output_weights.shape[1] == output_bias_weights.shape[0]

        def add_last_layer_equations():
            first_idx = self.network.getNumberOfVariables()
            self.network.setNumberOfVariables(first_idx + output_weights.shape[1])

            for i in range(output_weights.shape[1]):
                self.network.setLowerBound(i, -LARGE)
                self.network.setUpperBound(i, LARGE)
                eq = MarabouCore.Equation()
                for j, w in enumerate(output_weights[:, i]):
                    eq.addAddend(w, prev_layer_idx[j])
                eq.setScalar(-output_bias_weights[i])
                eq.addAddend(-1, first_idx + i)
                self.network.addEquation(eq)
            return first_idx, first_idx + output_weights.shape[1]

        def add_intermediate_layer_equations():
            first_idx = self.network.getNumberOfVariables()
            # times 2 for the b and f variables
            self.network.setNumberOfVariables(first_idx + (output_weights.shape[1] * 2))
            b_indices = range(first_idx, first_idx + (output_weights.shape[1] * 2), 2)
            f_indices = range(first_idx + 1, first_idx + (output_weights.shape[1] * 2), 2)
            for i in range(output_weights.shape[1]):
                cur_b_idx = b_indices[i]
                cur_f_idx = f_indices[i]
                # b variable
                self.network.setLowerBound(cur_b_idx, -LARGE)
                self.network.setUpperBound(cur_b_idx, LARGE)
                # f variable
                self.network.setLowerBound(cur_f_idx, 0)
                self.network.setUpperBound(cur_f_idx, LARGE)

                MarabouCore.addReluConstraint(self.network, cur_b_idx, cur_f_idx)
                # b equation
                eq = MarabouCore.Equation()
                for j, w in enumerate(output_weights[:, i]):
                    eq.addAddend(w, prev_layer_idx[j])
                eq.setScalar(-output_bias_weights[i])
                eq.addAddend(-1, cur_b_idx)
                self.network.addEquation(eq)
            return f_indices

        if layer.activation == tf.keras.activations.relu:
            # Internal layer
            return add_intermediate_layer_equations()
            # raise NotImplementedError("activation {} is not supported".format(layer.activation.__name__))
        elif layer.activation == tf.keras.activations.softmax or layer.activation == tf.keras.activations.linear:
            # last layer
            return add_last_layer_equations()
        else:
            raise NotImplementedError("activation {} is not supported".format(layer.activation.__name__))

    def set_input_bounds(self, xlim: list):
        '''
        set bounds on the input variables
        :param xlim: list of tuples each tuple is (lower_bound, upper_bound)
        '''
        assert len(xlim) == len(self.input_idx)
        for i, marabou_idx in enumerate(self.input_idx):
            self.network.setLowerBound(marabou_idx, xlim[i][0])
            self.network.setUpperBound(marabou_idx, xlim[i][1])

    def set_input_bounds_template(self, xlim: list, radius: float):
        '''
        set bounds on the input variables
        For example if xlim is [(5,3)], and radius is 0.1, the limit will be:
        0.9 * (5i + 3) <= x[0] <= 1.1 * (5i + 3)
        :param xlim: list of tuples, each tuple is (alpha, beta) which will be used as alpha * i + beta
        :param radius: non negative number, l_infinity around each of the points
        '''
        assert radius >= 0
        assert len(xlim) == len(self.input_idx)
        assert len(self._rnn_loop_idx) > 0
        u_r = 1 + radius  # upper radius
        l_r = 1 - radius  # lower radius
        i_idx = self._rnn_loop_idx[0]
        for i, marabou_idx in enumerate(self.input_idx):
            alpha, beta = xlim[i]
            self.network.setLowerBound(marabou_idx, -LARGE)
            self.network.setUpperBound(marabou_idx, LARGE)
            # TODO: What if alpha / beta == 0?
            # x <= r * alpha * i + r * beta <--> x - r * alpha * i <= r * beta
            ub_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
            ub_eq.addAddend(1, marabou_idx)
            ub_eq.addAddend(-u_r * alpha, i_idx)
            ub_eq.setScalar(u_r * beta)
            self.network.addEquation(ub_eq)

            # x >= r * alpha * i + r * beta <--> x - r * alpha * i >= r * beta
            lb_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
            lb_eq.addAddend(1, marabou_idx)
            lb_eq.addAddend(-l_r * alpha, i_idx)
            lb_eq.setScalar(l_r * beta)
            self.network.addEquation(lb_eq)


if __name__ == "__main__":
    r = RnnMarabouModel("models/model_20classes_rnn4_rnn2_fc16_epochs3.h5", 3)
    print(r.rnn_out_idx)
