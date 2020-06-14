import numpy as np
import torch

from maraboupy import MarabouCore
from RNN.MarabouRNNMultiDim import add_rnn_multidim_cells, prove_multidim_property, IterateAlphasSGD,

MODEL_FILE_PATH = 'maraboupy/mnist_example/rnn.pt'
large = 5000
small = 10 ** -4

MAX_SIZE = 4
ROUND = 4


class MnistModel():
    def __init__(self, model_file_path, perturbation_limit=0.1):
        self.state_dict = torch.load(model_file_path)
        self.perturbation_limit = perturbation_limit
        self.extract_rnn_weights()
        self.extract_output_weights()

        self.initial_values = None
        self.network = None

    def set_perturbation(self, new_limit):
        self.perturbation_limit = new_limit

    def extract_rnn_weights(self):
        '''
        rnn_input_weights.shape = [64, 112]
        rnn_bias_weights.shape = [64,]
        rnn_hidden_weights.shape = [64,64]
        :return: rnn_input_weights, rnn_hidden_weights, rnn_bias_weights
        '''
        self.w_in = self.state_dict['rnn.weight_ih_l0'].numpy()
        self.w_in = self.w_in[:MAX_SIZE, :MAX_SIZE]
        # self.b_in = self.state_dict['rnn.bias_ih_l0'].numpy()
        # self.b_in = self.b_in[:MAX_SIZE, :MAX_SIZE]
        self.w_h = self.state_dict['rnn.weight_hh_l0'].numpy()
        self.w_h = np.round(self.w_h, ROUND)
        self.w_h = self.w_h[:MAX_SIZE, :MAX_SIZE]
        # pytorch saves two biases but it's not more informative then having just one
        self.b_h = self.state_dict['rnn.bias_ih_l0'].numpy() + self.state_dict['rnn.bias_hh_l0'].numpy()
        self.b_h = np.round(self.b_h, ROUND)

        self.b_h = self.b_h[:MAX_SIZE]

        # return rnn_input_weights, rnn_hidden_weights, rnn_bias

    def extract_output_weights(self):
        # output_weights = self.state_dict['out.weight'].numpy()
        # output_bias_weights = self.state_dict['out.bias'].numpy()
        # return output_weights, output_bias_weights
        self.w_out = self.state_dict['out.weight'].numpy()
        self.b_out = self.state_dict['out.bias'].numpy()
        self.w_out = self.w_out[:MAX_SIZE, :MAX_SIZE]
        self.b_out = self.b_out[:MAX_SIZE]

    def _calc_output_initial_values(self, img_patch, n):
        assert img_patch.shape[0] == self.w_in.shape[1]  # 28 * 28 / 7 (where 7 is number of patches)
        x_min = img_patch * (1 - self.perturbation_limit)
        x_max = img_patch * (1 + self.perturbation_limit)

        r_min, r_max = calc_min_max_values(x_min, x_max, self.w_in, self.b_h)
        self.rnn_initial_values = [r_min, r_max]

        y_min, y_max = calc_min_max_values(r_min, r_max, self.w_out, self.b_out)
        self.initial_values = [y_min, y_max]
        # y, rs = get_max_value([x_min, x_max], self.w_in, self.w_h, self.b_h, self.w_out, self.b_out, n)
        print("After 1 iteration,, network max value: {}\nrecurrent max values: {}".format(y_max, r_max))
        # print("network max value:",
        #       get_max_value([x_min, x_max], self.w_in, self.w_h, self.b_h, self.w_out, self.b_out, n))
        return r_min, r_max

    def set_network_description(self, img_patch, n):
        if not self.initial_values:
            self._calc_output_initial_values(img_patch, n)
        self.network = MarabouCore.InputQuery()
        self.network.setNumberOfVariables(0)

        set_img_bounds(img_patch, self.network, self.perturbation_limit)
        self.rnn_output_idxs = add_rnn_cells(self.network, self.w_in, self.w_h, self.b_h, n)

        self.out_idx = add_output_equations(self.network, self.rnn_output_idxs, self.w_out, self.b_out)
        print("output idxs are", self.out_idx)
        # self.extract_output_weights()


    def prove_rnn_max_property(self, img_patch, rnn_out_idx, max_value, n):
        '''
        prove property on the rnn
        :param rnn_out_idx: one of rnn output idx
        :param max_value: max value for the output
        :param n: number of iterations
        :return:
        '''
        if img_patch is None:
            img_patch = np.array([0.1, 0.2, 0.3, 0.4] * 28) # 112
            # img_patch = np.array([0.2] * 112)
            # img_patch = np.load('1.pt')
        img_patch = img_patch[:MAX_SIZE]

        self.set_network_description(img_patch, n)

        property_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
        property_eq.addAddend(1, self.rnn_output_idxs[rnn_out_idx])
        property_eq.setScalar(max_value)
        rnn_start_idxs = [i - 3 for i in self.rnn_output_idxs]

        algorithm = IterateAlphasSGD(self.rnn_initial_values, rnn_start_idxs, self.rnn_output_idxs)
        return prove_multidim_property(self.network, rnn_start_idxs, self.rnn_output_idxs, [property_eq], algorithm)

    def prove_adv_property(self, img_patch, out_idx_max, out_idx_compare, n):
        '''
        prove property on the rnn
        :param img_path: The input img for the network
        :param out_idx_max: which index in the output should be maximum
        :param n: number of iterations
        :return:
        '''
        if img_patch is None:
            # img_patch = np.array([0.1, 0.2, 0.3, 0.4] * 28) # 112
            img_patch = np.array([0.2] * 112)
        img_patch = img_patch[:MAX_SIZE]


        properties = []
        self.set_network_description(img_patch, n)
        assert len(self.out_idx) > out_idx_max


        property_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
        property_eq.addAddend(1, self.out_idx[out_idx_max])
        property_eq.addAddend(-1, out_idx_compare)
        property_eq.setScalar(small)
        properties.append(property_eq)

        rnn_start_idxs = [i - 3 for i in self.rnn_output_idxs]
        return prove_multidim_property(self.network, rnn_start_idxs, self.rnn_output_idxs, self.rnn_initial_values,
                                       properties)

    def prove_out_max_property(self, out_idx, max_value):
        '''

        :param out_idx: index in the 10 vector output
        :param max_value: maximum value for that index
        :return: True / False
        '''
        raise NotImplementedError
        self._calc_output_initial_values(img_patch, n)
        property_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
        property_eq.addAddend(w_out[0], out_idx)
        property_eq.setScalar(max_value)

        assert prove_multidim_property(network, rnn_start_idxs, rnn_output_idxs, self.output_initial_values,
                                       [property_eq])


def set_img_bounds(img, network, pertubation_limit=0.05):
    '''

    :param img: ndarray with shape 128,  this is the input image
    :param network: marabou query that will add to it the limit on the input image
    :param pertubation_limit: how much pertubation is allwod in percepntage to the current value
    :return: int how many variables added to the network
    '''
    last_idx = network.getNumberOfVariables()
    network.setNumberOfVariables(last_idx + img.shape[-1])

    for i in range(img.shape[-1]):
        network.setLowerBound(last_idx + i, img[i] * (1 - pertubation_limit))
        network.setUpperBound(last_idx + i, img[i] * (1 + pertubation_limit))

    return img.shape[-1]


def add_rnn_cells(network, rnn_input_weights, rnn_hidden_weights, rnn_bias, num_iterations=10):
    '''
    adding n cells to the network, where n is the size of the hidden weights score matrix
    :param network: marabou query to add the cells
    :param rnn_input_weights: ndarray, shape n on m where m is the input size
    :param rnn_hidden_weights: square matrix, size is n on n
    :param rnn_bias: 1 dim array length n with the bias to the rnn cells
    :return: list of the output indcies for each of the cells
    '''
    assert type(rnn_input_weights) == np.ndarray
    assert type(rnn_hidden_weights) == np.ndarray and len(rnn_hidden_weights.shape) == 2
    assert rnn_hidden_weights.shape[0] == rnn_hidden_weights.shape[1]
    assert rnn_hidden_weights.shape[0] == rnn_input_weights.shape[0]
    assert len(rnn_bias) == rnn_hidden_weights.shape[0]

    var_weight_list = []

    # We assume here that the input variables index is 0 to rnn_input_weights.shape[0]
    # for i in range(rnn_input_weights.shape[0]):  # 64
    #     var_weight_list.append([(i, w) for w in rnn_input_weights[i, :]])
    input_indices = list(range(network.getNumberOfVariables()))
    rnn_output_idx = add_rnn_multidim_cells(network, input_indices, rnn_input_weights, rnn_hidden_weights, rnn_bias,
                                            num_iterations)
    # network.dump()
    return rnn_output_idx


def add_output_equations(network, rnn_output_idxs, output_weight, output_bias):
    '''
    build equations for the output
    :param network: network to append equations and variables to
    :param rnn_output_idxs: the output indices of the previous layer
    :param output_weight: Weights to multiply the previous layer
    :param output_bias: The bias of each equation
    :return: list of indices of output classes
    '''
    assert (len(rnn_output_idxs) == output_weight.shape[1])
    assert (output_weight.shape[0] == len(output_bias))
    last_idx = network.getNumberOfVariables()
    output_idxs = []
    network.setNumberOfVariables(last_idx + (output_weight.shape[0] * 2))  # *2 because of the relu
    for i in range(output_weight.shape[0]):
        b_variable_idx = last_idx + (2 * i)
        f_variable_idx = last_idx + 1 + (2 * i)
        output_idxs.append(f_variable_idx)

        network.setLowerBound(b_variable_idx, -large)
        network.setUpperBound(b_variable_idx, large)
        network.setLowerBound(f_variable_idx, 0)
        network.setUpperBound(f_variable_idx, large)
        MarabouCore.addReluConstraint(network, b_variable_idx, f_variable_idx)

        output_eq = MarabouCore.Equation()
        for j in range(output_weight.shape[1]):
            output_eq.addAddend(output_weight[i, j], rnn_output_idxs[j])

        output_eq.addAddend(-1, b_variable_idx)
        output_eq.setScalar(-output_bias[i])
        network.addEquation(output_eq)

    return output_idxs


def build_query(img_example: list, ylim: list, model: MnistModel):
    '''

    :param xlim: list of tuples, each cell is for input variable, and the tuple is (min_val, max_val)
    :param ylim: list of tuples, each cell is for an output variable, and the tuple is (min_val, max_val)
    :param pendulum_model: initialized model with get_rnn_weights and get_output_weights
    :return:
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(0)
    last_idx = set_img_bounds(img_example, network)

    rnn_input_weights, rnn_hidden_weights, rnn_bias = model.get_rnn_weights()
    rnn_output_idx = add_rnn_cells(network, rnn_input_weights, rnn_bias, rnn_hidden_weights)
    output_weights, output_bias_weights = model.get_output_weights()
    add_output_equations(network, rnn_output_idx, output_weights, output_bias_weights)
    return network


def or_in_realu(input_bounds, w_in, w_h, b_h, w_out, b_out):
    # TODO: Set of linear equations we can try to solve, probably will need bigger hidden layer (4x4 or something)
    r0, r1 = 0, 0

    r0_0 = input_bounds[1] * w_in[0] + r0 * w_h[0, 0] + r1 * w_h[0, 1] + b_h[0]
    r1_0 = input_bounds[1] * w_in[1] + r0 * w_h[1, 0] + r1 * w_h[1, 1] + b_h[1]

    r0_1 = input_bounds[1] * w_in[0] + r0_0 * w_h[0, 0] + r1_0 * w_h[0, 1] + b_h[0]
    r1_1 = input_bounds[1] * w_in[1] + r0_0 * w_h[1, 0] + r1_0 * w_h[1, 1] + b_h[1]

    r0_2 = input_bounds[1] * w_in[0] + r0_1 * w_h[0, 0] + r1_1 * w_h[0, 1] + b_h[0]
    r1_2 = input_bounds[1] * w_in[1] + r0_1 * w_h[1, 0] + r1_1 * w_h[1, 1] + b_h[1]

    out_0_actual = r0_0 * w_out[0] + b_out[0] + r1_0 * w_out[1] + b_out[1]
    out_1_actual = r0_1 * w_out[0] + b_out[0] + r1_1 * w_out[1] + b_out[1]
    out_2_actual = r0_2 * w_out[0] + b_out[0] + r1_2 * w_out[1] + b_out[1]

    out_0 = r0
    out_1 = r0 or r1
    out_2 = r0 and r1

    assert out_0_actual == out_0
    assert out_1_actual == out_1
    assert out_2_actual == out_2


def get_max_value(input_bounds, w_in, w_h, b_h, w_out, b_out, n):
    '''
    calculate the max value of a network with one input node, 2d rnn cell, and a number
    :param input_bounds: tuple (min_val, max_val)
    :param w_in: weights on the input to the rnn node array length 2
    :param w_h: hidden matrix, 2x2
    :param b_h: bias for rnn cell array length 2
    :param w_out: weight from the rnn cell to the output array length 2
    :param b_out: bias from the rnn cell to the output length 2
    :param n: number of rounds
    :return: max value of the
    '''
    r0 = 0
    r1 = 1
    r = np.zeros(w_h.shape[0])
    rs = []
    for i in range(n):
        r_new = ReLU(np.matmul(input_bounds[1], w_in.T) + np.matmul(r, w_h) + b_h[0])
        rs.append(r_new)
        r = r_new
        # r0_new = input_bounds[1] * w_in[0] + r0 * w_h[0,0] + r1 * w_h[0,1] + b_h[0]
        # r1_new = input_bounds[1] * w_in[1] + r0 * w_h[1, 0] + r1 * w_h[1, 1] + b_h[1]
        # r0 = r0_new
        # r1 = r1_new
    assert np.vectorize(lambda x: x >= 0)(r).all()
    y = np.matmul(r, w_out.T) + b_out
    return y, rs


def ReLU(x):
    '''
    return element wise ReLU (max between 0,x)
    :param x: int or np.ndarray
    :return: same type, only positive numbers
    '''
    if isinstance(x, int):
        return max(0, x)
    elif isinstance(x, np.ndarray):
        return np.array(list(map(lambda v: max(0, v), x)))
    else:
        return None


def calc_min_max_values(min_input, max_input, w_in, b_h):
    '''
    Calc the first min and max values, the hidden memory is zero
    assume there are n inputs
    :param min_input: vector size n of minimum bound for the input
    :param max_input: vector size n of maximum bound for the input
    :param w_in: weight matrix, n*m, where m is the hidden layer dimension
    :return: tuple (min values, max_values), each is vector length m
    '''
    r_min = np.zeros(w_in.shape[0])
    r_max = np.zeros(w_in.shape[0])
    for i in range(w_in.shape[0]):
        for j in range(w_in.shape[1]):
            w = w_in[i, j]
            if w > 0:
                r_min[i] += min_input[j] * w
                r_max[i] += max_input[j] * w
            else:
                r_min[i] += max_input[j] * w
                r_max[i] += min_input[j] * w

    r_min = ReLU(np.array(r_min) + b_h)
    r_max = ReLU(np.array(r_max) + b_h)
    r_min_old = ReLU(np.matmul(min_input, w_in.T) + b_h)
    r_max_old = ReLU(np.matmul(max_input, w_in.T) + b_h)
    assert (r_max >= r_max_old).all() & (r_min <= r_min_old).all()
    return r_min, r_max


def prove_rnn_max_property():
    model = MnistModel(MODEL_FILE_PATH)
    if model.prove_rnn_max_property(None, 1, 0.013, 6):
        print("property proved")


def prove_output_adv():
    model = MnistModel(MODEL_FILE_PATH)
    out_idx_max = 1
    for i in range(10):
        if i != 3:
            if model.prove_adv_property(None, out_idx_max, i, 2):
                print("proved that 3 and not", i)


def boundEqConflict():
    '''
    Simple presecion exmaple.
    Only two nodes that are conncted with ReLU, and an equation that asks if the ReLU output is very small negative
    :return:
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(2)

    network.setLowerBound(0, -5)
    network.setUpperBound(0, 5)

    network.setLowerBound(1, 0)
    network.setUpperBound(1, 5)

    MarabouCore.addReluConstraint(network, 0, 1)

    eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    eq.addAddend(1, 1)
    eq.setScalar(-10**-4) # -10 ** -4 works
    network.addEquation(eq)

    verbose = 2
    vars1, stats1 = MarabouCore.solve(network, "", 0, verbose)
    if len(vars1) > 0:
        print("SAT")
        print(vars1)
        return False
    else:
        print("UNSAT")
        return True


if __name__ == "__main__":
    np.random.seed(0)
    prove_rnn_max_property()
    exit(0)

    n = 10
    img = np.array([1])  # img = np.array([1, 1])  # np.random.random(2)
    w_in_0 = [1]  # w_in_0 = [1, 1]
    w_in_1 = [1]  # w_in_1 = [1, 1]

    w_h_0 = [1, 1]
    w_h_1 = [0, 0]
    b_h = [0, 0]
    w_out_0 = [1]
    w_out_1 = [1]

    # w_in_0 = [0.1, 0.2]
    # w_in_1 = [0.3, 0.4]
    # w_h_0 = [0.5, 0.7]
    # w_h_1 = [0.2, 0.3]
    # w_out_0 = [0.2]
    # w_out_1 = [1.2]

    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(0)

    pertubation_limit = 0.1
    set_img_bounds(img, network, pertubation_limit)

    x_min = img * (1 - pertubation_limit)
    x_max = img * (1 + pertubation_limit)

    w_in = np.array([w_in_0, w_in_1])
    w_h = np.array([w_h_0, w_h_1])

    rnn_output_idxs = add_rnn_cells(network, w_in, w_h, b_h, n)
    rnn_start_idxs = [i - 3 for i in rnn_output_idxs]

    w_out = np.array([1, 0])
    property_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    property_eq.addAddend(w_out[0], rnn_output_idxs[0])
    property_eq.setScalar(23.1)

    # draw_r_values(*calc_rnn_values(x_max, w_in, w_h, n))
    w_in = np.array([w_in_0, w_in_1])
    r_min, r_max = calc_min_max_values(x_min, x_max, w_in, np.zeros(w_in.shape))
    initial_values = [r_min, r_max]
    y, rs = get_max_value([x_min, x_max], w_in, w_h, b_h, w_out, 0, n)
    print("network max value: {}\nrecurrent max values: {}".format(y, rs))
    assert prove_multidim_property(network, rnn_start_idxs, rnn_output_idxs, initial_values, [property_eq])
    print('property proved')
    #
    # # print("rnn_start_idxs:", rnn_start_idxs)
    # output_idx = add_output_equations(network, rnn_output_idxs, np.array([w_out_0, w_out_1]).T, np.array([0.3]))
    # # network.dump()
    # # print("output idx:", output_idx)
    #
    # # Let's verify that the output node is less then 100, after 10 iterations
    # n = 10
    # property_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    # property_eq.addAddend(1, output_idx[0])
    # property_eq.setScalar(100)
    #
    # # network, rnn_start_idxs, rnn_invariant_type, initial_values, n_iterations,
    # # property_equations, min_alphas = None, max_alphas = None, rnn_dependent = None
    #
    # find_invariant_marabou(network, rnn_start_idxs, [MarabouCore.Equation.LE] * len(rnn_start_idxs), initial_values,  n, [property_eq])

    # model = MnistModel(MODEL_FILE_PATH)
    # build_query(np.random.random((128)), None, model)
