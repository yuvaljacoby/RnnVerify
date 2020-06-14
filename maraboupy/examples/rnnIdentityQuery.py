from maraboupy import MarabouCore


# Simple RNN, similar to MarabouCoreExample but gets list of inputs
#   0   <= xi  <= 1
#   0   <= sif          # rnn cell
#   0   <= zif          # rnn cell
#   1/2 <= y  <= 1
#
# Equations:
#  x1 - s1b = 0
#  x1 + z1b = 0
#  for each i > 1
#       xi + s(i-1)f - sib = 0
#       xi - z(i-1)f - zib = 0
#  y - skf - zkf =0 # Where k == num_iterations
#
#  sif = Relu(sib)
#
# Parameters:
#   from - to
#   x1 - x(i-1): input
#   xi - x(3*i): (alternating between sib and sif)
#       xi:          sib
#       x(i+1):      sif
#   x(3*i) - x(5*i): (alternating between zib and zif)
#       xi:          zib
#       x(i+1):      zif
#   x(5i):       output


def add_rnn_cell_bounds(inputQuery, num_iterations, i, large):
    '''
    add constraints for rnn hidden vector (unfolded)
    for each hidden vector add b and f constraint
    the constraint are for each hidden vector i:
        constraint i: between -large to large (unbounded)
        constraint i + 1: between 0 to large ReLu result
    :param inputQuery query to add the bounds too
    :param num_iterations: number of hidden vectors (will add two for each constraint)
    :param i: start index
    :param large: big number
    :return: update i
    '''
    for _ in range(num_iterations):
        # sib
        inputQuery.setLowerBound(i, -large)
        inputQuery.setUpperBound(i, large)

        # sif
        inputQuery.setLowerBound(i + 1, 0)
        inputQuery.setUpperBound(i + 1, large)
        i += 2
    return i


large = 10.0
num_iterations = 3
i = 0  # index for variable number

inputQuery = MarabouCore.InputQuery()

num_variables = num_iterations  # the x input
s_first_index = num_variables
num_variables += num_iterations * 2  # for each s temporal state (2 because of the ReLu)
z_first_index = num_variables
num_variables += num_iterations * 2  # for each z temporal state (2 because of the ReLu)
y_index = num_variables
num_variables += 1  # for y

inputQuery.setNumberOfVariables(num_variables)

# set x variables bounds
for _ in range(num_iterations):
    inputQuery.setLowerBound(i, 0)
    inputQuery.setUpperBound(i, 1)
    i += 1

i = add_rnn_cell_bounds(inputQuery, num_iterations, i, large)  # add s_i

add_rnn_cell_bounds(inputQuery, num_iterations, i, large)  # add z_i

# output
inputQuery.setLowerBound(i, 0.5)
inputQuery.setUpperBound(i, 0.9)

# x1 - s1b = 0
equation1 = MarabouCore.Equation()
equation1.addAddend(1, 0)
equation1.addAddend(-1, s_first_index)
equation1.setScalar(0)
inputQuery.addEquation(equation1)

# x1 - z1b = 0
equation1 = MarabouCore.Equation()
equation1.addAddend(1, 0)
equation1.addAddend(-1, z_first_index)
equation1.setScalar(0)
inputQuery.addEquation(equation1)


def add_hidden_state_equations(inputQuery, variables_first_index, input_weight, hidden_weight, num_iterations):
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
    :param num_iterations: number of iterations
    :return:
    '''
    equation1 = MarabouCore.Equation()
    equation1.addAddend(input_weight, 0)
    equation1.addAddend(-1, variables_first_index)
    equation1.setScalar(0)
    inputQuery.addEquation(equation1)

    for k in range(1, num_iterations):
        cur_equation = MarabouCore.Equation()
        cur_equation.addAddend(input_weight, k)  # xk
        cur_equation.addAddend(hidden_weight, variables_first_index + (2 * k) - 1)  # s(k-1)f
        cur_equation.addAddend(-1, variables_first_index + (2 * k))  # skb
        cur_equation.setScalar(0)
        inputQuery.addEquation(cur_equation)

    # ReLu's
    for k in range(variables_first_index, variables_first_index + 2 * num_iterations, 2):
        MarabouCore.addReluConstraint(inputQuery, k, k + 1)


add_hidden_state_equations(inputQuery, s_first_index, 1, 1, num_iterations)
add_hidden_state_equations(inputQuery, z_first_index, 1, -1, num_iterations)

# y - skf - zkf = 0
output_equation = MarabouCore.Equation()
output_equation.addAddend(1, y_index)
output_equation.addAddend(-1, z_first_index - 1)
output_equation.addAddend(-1, y_index - 1)
output_equation.setScalar(0)
inputQuery.addEquation(output_equation)


vars1, stats1 = MarabouCore.solve(inputQuery, "", 0)
if len(vars1) > 0:
    print("SAT")
    print(vars1)
else:
    print("UNSAT")
