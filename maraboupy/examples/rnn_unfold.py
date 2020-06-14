import time

from maraboupy import MarabouCore

# Most simple RNN, only sums inputs
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
#
# Parameters:
#   from - to
#   x1 - x(i-1): input
#   xi - x(3*i): (alternating between sib and sif)
#   xi:          sib
#   x(i+1):      sif
#   x(3i):       output


large = 1000.0
small = 10 ** -2


def add_rnn_cell_bounds(inputQuery, n_iterations, i, large):
    '''
    add constraints for rnn hidden vector (unfolded)
    for each hidden vector add b and f constraint
    the constraint are for each hidden vector i:
        constraint i: between -large to large (unbounded)
        constraint i + 1: between 0 to large ReLu result
    :param inputQuery query to add the bounds too
    :param n_iterations: number of hidden vectors (will add two for each constraint)
    :param i: start index
    :param large: big number
    :return: update i
    '''
    for _ in range(n_iterations):
        # sib
        inputQuery.setLowerBound(i, -large)
        inputQuery.setUpperBound(i, large)

        # sif
        inputQuery.setLowerBound(i + 1, 0)
        inputQuery.setUpperBound(i + 1, large)
        i += 2
    return i


def add_hidden_state_equations(inputQuery, variables_first_index, input_weight, hidden_weight, n_iterations):
    '''
    add all hidden state equations:
        input_weight * x1 = s1b
        for each k > 1
            input_weight * xi + hidden_weight * s(k-1)f = sib
        and ReLu's
    :param inputQuery: query to append to
    :param variables_first_index: the first index of the hidden vector variable
    :param input_weight: the weight in the input
    :param hidden_weight: the weight for the hidden vector
    :param n_iterations: number of iterations
    :return:
    '''
    equation1 = MarabouCore.Equation()
    equation1.addAddend(input_weight, 0)
    equation1.addAddend(-1, variables_first_index)
    equation1.setScalar(0)
    inputQuery.addEquation(equation1)

    for k in range(1, n_iterations):
        cur_equation = MarabouCore.Equation()
        cur_equation.addAddend(input_weight, k)  # xk
        cur_equation.addAddend(hidden_weight, variables_first_index + (2 * k) - 1)  # s(k-1)f
        cur_equation.addAddend(-1, variables_first_index + (2 * k))  # skb
        cur_equation.setScalar(0)
        inputQuery.addEquation(cur_equation)

    # ReLu's
    for k in range(variables_first_index, variables_first_index + 2 * n_iterations, 2):
        MarabouCore.addReluConstraint(inputQuery, k, k + 1)


def unfold_sum_rnn(n_iterations, xlim=(-1, 1), ylim=(-1, 1)):
    i = 0  # index for variable number
    inputQuery = MarabouCore.InputQuery()

    num_variables = n_iterations  # the x input
    s_first_index = num_variables
    num_variables += n_iterations * 2  # for each temporal state (2 because of the ReLu)
    y_index = num_variables
    num_variables += 1  # for y

    inputQuery.setNumberOfVariables(num_variables)

    for _ in range(n_iterations):
        inputQuery.setLowerBound(i, xlim[0])
        inputQuery.setUpperBound(i, xlim[1])
        i += 1

    add_rnn_cell_bounds(inputQuery, n_iterations, s_first_index, large)  # add s_i

    # output
    inputQuery.setLowerBound(y_index, ylim[0])
    inputQuery.setUpperBound(y_index, ylim[1])

    add_hidden_state_equations(inputQuery, s_first_index, 1, 1, n_iterations)

    # y - skf = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, y_index)
    output_equation.addAddend(-1, y_index - 1)
    output_equation.setScalar(0)
    inputQuery.addEquation(output_equation)

    vars1, stats1 = MarabouCore.solve(inputQuery, "", 0)
    if len(vars1) > 0:
        print("SAT")
        print(vars1)
    else:
        print("UNSAT")


if __name__ == "__main__":
    num_iterations = 4
    xlim = (-1, 1)
    pass_run = True
    if pass_run:
        ylim = (xlim[1] * num_iterations + 0.1, xlim[1] * num_iterations + 1)
    else:
        ylim = (xlim[0] * num_iterations, xlim[1] * num_iterations)

    start = time.time()
    unfold_sum_rnn(num_iterations, xlim, ylim)
    end = time.time()
    print('took', round(end - start, 3), 'seconds')
