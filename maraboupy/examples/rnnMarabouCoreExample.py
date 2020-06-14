import time

from z3 import *
from maraboupy import MarabouCore



large = 1000.0
small = 10 ** -2


def marabou_solve_negate_eq(query):
    vars1, stats1 = MarabouCore.solve(query, "", 0)
    if len(vars1) > 0:
        print("SAT")
        print(vars1)
        return False
    else:
        print("UNSAT")
        return True


# <editor-fold desc="negative sum network definition">

# Most simple RNN, only sums the positive inputs
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
def define_negative_sum_output_equations(network, ylim, n_iterations):
    '''
    defines the equations for validating the wanted property.
    Changes the query according (if needed)
    :param network: marabou definition of the positive_sum network
    :param ylim: ensure the output of the network is not more than ylim
    :param n_iterations: number of iterations that the network should run (maximum)
    :return: list of equations to validate the property)
    '''
    start_param = network.getNumberOfVariables()
    network.setNumberOfVariables(start_param + 1)
    network.setLowerBound(start_param, 0)
    network.setUpperBound(start_param, n_iterations)

    # make sure the property hold i.e. y <= ylim
    # we negate that and hope to get UNSAT i.e.
    # we want to check that y > ylim <--> y >= ylim + epsilon
    property_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    property_eq.addAddend(1, 4)
    # property_eq.addAddend(-weight, 1)
    property_eq.setScalar(ylim[1] + small)
    return [property_eq]


def define_negative_sum_invariant_equations(query):
    '''
    Define the equations for invariant, if needs more params should update the query with them
    and we need to define it in the calling function (not the best way but some
    :param query: marabou definition of the positive_sum network, will be changed if needed
    :return: tuple ([base equations], [step equations], [equations that hold if invariant hold])
    '''
    start_param = query.getNumberOfVariables()
    query.setNumberOfVariables(start_param + 1)

    # Add the slack variable, i
    query.setLowerBound(start_param, 0)
    query.setUpperBound(start_param, large)

    # (s_0 f) = 0
    base_hidden_limit_eq = MarabouCore.Equation()
    base_hidden_limit_eq.addAddend(1, 1)
    base_hidden_limit_eq.setScalar(0)

    # (s_i-1 f) <= i - 1 <--> i - (s_i-1 f) >= 1
    hidden_limit_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    hidden_limit_eq.addAddend(1, start_param)  # i
    hidden_limit_eq.addAddend(-1, 1)  # s_i-1 f
    hidden_limit_eq.setScalar(1)
    # query.addEquation(hidden_limit_eq)

    # negate the invariant we want to prove
    # not(s_1 b <= 1) <--> s_1 b  > 1  <--> s_1 b >= 1 + \epsilon
    base_output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    base_output_equation.addAddend(1, 2)
    base_output_equation.setScalar(1 + small)

    # not (s_i b >= i) <--> s_i b < i <--> s_i b -i >= \epsilon
    output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    output_equation.addAddend(1, 2)  # s_i b
    output_equation.addAddend(-1, start_param)  # i
    output_equation.setScalar(small)

    # s_i b <= i
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, 2)  # s_i b
    invariant_equation.addAddend(-1, start_param)  # i
    invariant_equation.setScalar(0)

    base_invariant_eq = [base_hidden_limit_eq, base_output_equation]
    step_invariant_eq = [hidden_limit_eq, output_equation]
    return (base_invariant_eq, step_invariant_eq, [invariant_equation])


def define_negative_sum_network(xlim=(-1, 1)):
    '''
    Defines the positive_sum network in a marabou way, without the recurrent part
    i.e. we define:
        s_i b = s_i-1 f + x_i
        y = s_i f
    :param xlim: how to limit the input to the network
    :return: query to marabou that defines the positive_sum rnn network (without recurent)
    '''
    num_params_for_cell = 5

    positive_sum_rnn_query = MarabouCore.InputQuery()
    positive_sum_rnn_query.setNumberOfVariables(num_params_for_cell)

    # x
    positive_sum_rnn_query.setLowerBound(0, xlim[0])
    positive_sum_rnn_query.setUpperBound(0, xlim[1])

    # s_i-1 f (or temp in some of my notes)
    positive_sum_rnn_query.setLowerBound(1, 0)
    positive_sum_rnn_query.setUpperBound(1, large)

    # s_i b
    positive_sum_rnn_query.setLowerBound(2, -large)
    positive_sum_rnn_query.setUpperBound(2, large)

    # s_i f
    positive_sum_rnn_query.setLowerBound(3, 0)
    positive_sum_rnn_query.setUpperBound(3, large)

    # y
    positive_sum_rnn_query.setLowerBound(4, -large)
    positive_sum_rnn_query.setUpperBound(4, large)

    # s_i b = -x_i * 1 + s_i-1 f * 1
    update_eq = MarabouCore.Equation()
    update_eq.addAddend(-1, 0)
    update_eq.addAddend(1, 1)
    update_eq.addAddend(-1, 2)
    update_eq.setScalar(0)
    # update_eq.dump()
    positive_sum_rnn_query.addEquation(update_eq)

    # s_i f = ReLu(s_i b)
    MarabouCore.addReluConstraint(positive_sum_rnn_query, 2, 3)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, 4)
    output_equation.addAddend(-1, 3)
    output_equation.setScalar(0)
    # output_equation.dump()
    positive_sum_rnn_query.addEquation(output_equation)

    return positive_sum_rnn_query

# </editor-fold>

#<editor-fold desc="positive sum network defention">

# Most simple RNN, only sums the positive inputs
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
def define_positive_sum_output_equations(network, ylim, n_iterations):
    '''
    defines the equations for validating the wanted property.
    Changes the query according (if needed)
    :param network: marabou definition of the positive_sum network
    :param ylim: ensure the output of the network is not more than ylim
    :param n_iterations: number of iterations that the network should run (maximum)
    :return: list of equations to validate the property)
    '''

    start_param = network.getNumberOfVariables()
    network.setNumberOfVariables(start_param + 1)
    network.setLowerBound(start_param, 0)
    network.setUpperBound(start_param, n_iterations)

    # make sure the property hold i.e. y <= ylim
    # we negate that and hope to get UNSAT i.e.
    # we want to check that y > ylim <--> y >= ylim + epsilon
    property_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    property_eq.addAddend(1, 4)
    # property_eq.addAddend(-weight, 1)
    property_eq.setScalar(ylim[1] + small)
    return [property_eq]


def define_positive_sum_invariant_equations(query):
    '''
    Define the equations for invariant, if needs more params should update the query with them
    and we need to define it in the calling function (not the best way but some
    :param query: marabou definition of the positive_sum network, will be changed if needed
    :return: tuple ([base equations], [step equations], [equations that hold if invariant hold])
    '''
    start_param = query.getNumberOfVariables()
    query.setNumberOfVariables(start_param + 1)

    # Add the slack variable, i
    query.setLowerBound(start_param, 0)
    query.setUpperBound(start_param, large)

    # (s_0 f) = 0
    base_hidden_limit_eq = MarabouCore.Equation()
    base_hidden_limit_eq.addAddend(1, 1)
    base_hidden_limit_eq.setScalar(0)

    # (s_i-1 f) <= i - 1 <--> i - (s_i-1 f) >= 1
    hidden_limit_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    hidden_limit_eq.addAddend(1, start_param)  # i
    hidden_limit_eq.addAddend(-1, 1)  # s_i-1 f
    hidden_limit_eq.setScalar(1)
    # query.addEquation(hidden_limit_eq)

    # negate the invariant we want to prove
    # not(s_1 b <= 1) <--> s_1 b  > 1  <--> s_1 b >= 1 + \epsilon
    base_output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    base_output_equation.addAddend(1, 2)
    base_output_equation.setScalar(1 + small)

    # not (s_i b >= i) <--> s_i b < i <--> s_i b -i >= \epsilon
    output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    output_equation.addAddend(1, 2)  # s_i b
    output_equation.addAddend(-1, start_param)  # i
    output_equation.setScalar(small)

    # s_i b <= i
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, 2)  # s_i b
    invariant_equation.addAddend(-1, start_param)  # i
    invariant_equation.setScalar(0)

    base_invariant_eq = [base_hidden_limit_eq, base_output_equation]
    step_invariant_eq = [hidden_limit_eq, output_equation]
    return (base_invariant_eq, step_invariant_eq, [invariant_equation])


def define_positive_sum_network(xlim=(-1, 1)):
    '''
    Defines the positive_sum network in a marabou way, without the recurrent part
    i.e. we define:
        s_i b = s_i-1 f + x_i
        y = s_i f
    :param xlim: how to limit the input to the network
    :return: query to marabou that defines the positive_sum rnn network (without recurent)
    '''
    num_params_for_cell = 5

    # Plus one is for the invariant proof, we will add a slack variable
    positive_sum_rnn_query = MarabouCore.InputQuery()
    positive_sum_rnn_query.setNumberOfVariables(num_params_for_cell)  # + extra_params)

    # x
    positive_sum_rnn_query.setLowerBound(0, xlim[0])
    positive_sum_rnn_query.setUpperBound(0, xlim[1])

    # s_i-1 f (or temp in some of my notes)
    positive_sum_rnn_query.setLowerBound(1, 0)
    positive_sum_rnn_query.setUpperBound(1, large)

    # s_i b
    positive_sum_rnn_query.setLowerBound(2, -large)
    positive_sum_rnn_query.setUpperBound(2, large)

    # s_i f
    positive_sum_rnn_query.setLowerBound(3, 0)
    positive_sum_rnn_query.setUpperBound(3, large)

    # y
    positive_sum_rnn_query.setLowerBound(4, -large)
    positive_sum_rnn_query.setUpperBound(4, large)

    # s_i b = x_i * 1 + s_i-1 f * 1
    update_eq = MarabouCore.Equation()
    update_eq.addAddend(1, 0)
    update_eq.addAddend(1, 1)
    update_eq.addAddend(-1, 2)
    update_eq.setScalar(0)
    # update_eq.dump()
    positive_sum_rnn_query.addEquation(update_eq)

    # s_i f = ReLu(s_i b)
    MarabouCore.addReluConstraint(positive_sum_rnn_query, 2, 3)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, 4)
    output_equation.addAddend(-1, 3)
    output_equation.setScalar(0)
    # output_equation.dump()
    positive_sum_rnn_query.addEquation(output_equation)

    return positive_sum_rnn_query

#</editor-fold>

#<editor-fold desc="defention of network that returns last input">

# Most simple RNN, only sums the positive inputs
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
def define_last_output_equations(network, ylim, n_iterations):
    '''
    defines the equations for validating the wanted property.
    Changes the query according (if needed)
    :param network: marabou definition of the positive_sum network
    :param ylim: ensure the output of the network is not more than ylim
    :param n_iterations: number of iterations that the network should run (maximum)
    :return: list of equations to validate the property)
    '''


    # make sure the property hold i.e. y <= ylim
    # we negate that and hope to get UNSAT i.e.
    # we want to check that y > ylim <--> y >= ylim + epsilon
    property_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    property_eq.addAddend(1, 4)
    # property_eq.addAddend(-weight, 1)
    property_eq.setScalar(ylim[1] + small)
    return [property_eq]


def define_last_invariant_equations(query):
    '''
    Define the equations for invariant, if needs more params should update the query with them
    and we need to define it in the calling function (not the best way but some
    :param query: marabou definition of the positive_sum network, will be changed if needed
    :return: tuple ([base equations], [step equations], [equations that hold if invariant hold])
    '''

    # (s_0 f) = 0
    base_hidden_limit_eq = MarabouCore.Equation()
    base_hidden_limit_eq.addAddend(1, 1)
    base_hidden_limit_eq.setScalar(0)

    # (s_i-1 f) <= xlim
    xlim = (query.getLowerBound(0), query.getUpperBound(0))

    hidden_limit_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    hidden_limit_eq.addAddend(-1, 1)  # s_i-1 f
    hidden_limit_eq.setScalar(xlim[1])

    # negate the invariant we want to prove
    # not(s_1 b <= xlim[1]) <--> s_1 b  > xlim[1]  <--> s_1 b >= xlim[1] + \epsilon
    base_output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    base_output_equation.addAddend(1, 2)
    base_output_equation.setScalar(xlim[1] + small)

    # not (s_i b >= xlim[1]) <--> s_i b < xlim[1] <--> s_i b >= xlim[1] + \epsilon
    output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    output_equation.addAddend(1, 2)  # s_i b
    output_equation.setScalar(xlim[1] + small)

    # s_i b <= xlim[1]
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, 2)  # s_i b
    invariant_equation.setScalar(xlim[1])

    base_invariant_eq = [base_hidden_limit_eq, base_output_equation]
    step_invariant_eq = [hidden_limit_eq, output_equation]
    return (base_invariant_eq, step_invariant_eq, [invariant_equation])


def define_last_network(xlim=(-1, 1)):
    '''
    Defines the positive_sum network in a marabou way, without the recurrent part
    i.e. we define:
        s_i b = s_i-1 f + x_i
        y = s_i f
    :param xlim: how to limit the input to the network
    :return: query to marabou that defines the positive_sum rnn network (without recurent)
    '''
    num_params_for_cell = 5

    query = MarabouCore.InputQuery()
    query.setNumberOfVariables(num_params_for_cell)

    # x
    query.setLowerBound(0, xlim[0])
    query.setUpperBound(0, xlim[1])

    # s_i-1 f (or temp in some of my notes)
    query.setLowerBound(1, 0)
    query.setUpperBound(1, large)

    # s_i b
    query.setLowerBound(2, -large)
    query.setUpperBound(2, large)

    # s_i f
    query.setLowerBound(3, 0)
    query.setUpperBound(3, large)

    # y
    query.setLowerBound(4, -large)
    query.setUpperBound(4, large)

    # s_i b = x_i * 1
    update_eq = MarabouCore.Equation()
    update_eq.addAddend(1, 0)
    # update_eq.addAddend(1, 1)
    update_eq.addAddend(-1, 2)
    update_eq.setScalar(0)
    # update_eq.dump()
    query.addEquation(update_eq)

    # s_i f = ReLu(s_i b)
    MarabouCore.addReluConstraint(query, 2, 3)

    # y - skf  = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, 4)
    output_equation.addAddend(-1, 3)
    output_equation.setScalar(0)
    # output_equation.dump()
    query.addEquation(output_equation)

    return query

#</editor-fold>

#<editor-fold desc="sum network defention">

# simple RNN, only sums all inputs
#   0   <= xi  <= 1
#   0   <= sif
#   0   <= zif
#   1/2 <= y  <= 1
#
# Equations:
#  x1 - s1b = 0
#  x1 + z1b = 0
#  for each i > 1
#       xi + s(i-1)f - sib = 0
#  for each i > 1
#       -xi + z(i-1)f - zib = 0
#  y - skf - zkf =0 # Where k == n_iterations
#
#  sif = Relu(sib)
#  zxif = Relu(zib)
def define_sum_max_output_equations(network, ylim, n_iterations):
    '''
    defines the equations for validating the wanted property.
    Changes the query according (if needed)
    :param network: marabou definition of the sum network
    :param ylim: ensure the output of the network is exactly ylim
    :param n_iterations: number of iterations that the network should run (maximum)
    :return: list of equations to validate the property)
    '''
    start_param = network.getNumberOfVariables()
    network.setNumberOfVariables(start_param + 1)
    network.setLowerBound(start_param, 0)
    network.setUpperBound(start_param, n_iterations)

    # make sure the property hold i.e. y <= ylim
    # we negate that and hope to get UNSAT i.e.
    # we want to check that y <= ylim <--> y >= ylim + \epsilon
    property_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    property_eq.addAddend(1, 4)
    property_eq.setScalar(ylim[1] + small)
    return [property_eq]


def define_sum_invariant_max_equations(query):
    '''
    Define the equations for invariant, if needs more params should update the query with them
    and we need to define it in the calling function (not the best way but some
    :param query: marabou definition of the sum network, will be changed if needed
    :return: tuple ([base equations], [step equations], [equations that hold if invariant hold])
    '''
    xlim = (query.getLowerBound(0), query.getUpperBound(0))
    start_param = query.getNumberOfVariables()
    query.setNumberOfVariables(start_param + 1)

    # Add the slack variable, i
    query.setLowerBound(start_param, 0)
    query.setUpperBound(start_param, large)

    # (s_0 f) = 0
    base_hidden_limit_eq_s = MarabouCore.Equation()
    base_hidden_limit_eq_s.addAddend(1, 1)
    base_hidden_limit_eq_s.setScalar(0)

    # (z_0 f) = 0
    base_hidden_limit_eq_z = MarabouCore.Equation()
    base_hidden_limit_eq_z.addAddend(1, 4)
    base_hidden_limit_eq_z.setScalar(0)

    # negate the invariant we want to prove
    # not(s_1 b + z_1 b <= xlim[1]) <--> s_1 b + z_1 b  > xlim[1] <--> s_1 b + z_1 b >= xlim[1] + \epsilon
    base_output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    base_output_equation.addAddend(1, 2)
    base_output_equation.addAddend(1, 5)
    base_output_equation.setScalar(xlim[1] + small)

    # TODO: Add also GE from 1 + small and somehow validate also that

    # (s_i-1 f) + (z_i-1 f) <= xlim[1] * (i - 1) <--> (s_i-1 f) + (z_i-1 f) - i * xlim[1] <= -xlim[1]
    hidden_limit_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    hidden_limit_eq.addAddend(-xlim[1], start_param)  # i
    hidden_limit_eq.addAddend(1, 1)  # s_i-1 f
    hidden_limit_eq.addAddend(1, 4)  # z_i-1 f
    hidden_limit_eq.setScalar(-xlim[1])

    # not(s_i b + z_i b <= xlim[1] * i) <--> s_i b + z_i b  > xlim[1] * i <--> s_i b + z_i b - xlim[1] * i >= \epsilon
    output_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    output_equation.addAddend(1, 2)  # s_i b
    output_equation.addAddend(1, 5)  # z_i b
    output_equation.addAddend(-xlim[1], start_param)  # i
    output_equation.setScalar(-small)
    # TODO: Add also GE from 1 + small and somehow validate also that

    # s_i b + z_i b <= xlim[1] * i
    invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    invariant_equation.addAddend(1, 2)  # s_i b
    invariant_equation.addAddend(1, 5)  # z_i b
    invariant_equation.addAddend(-xlim[1], start_param)  # i
    invariant_equation.setScalar(0)

    base_invariant_eq = [base_hidden_limit_eq_s, base_hidden_limit_eq_z, base_output_equation]
    step_invariant_eq = [hidden_limit_eq, output_equation]
    return (base_invariant_eq, step_invariant_eq, [invariant_equation])


def define_sum_network(xlim=(-1, 1)):
    '''
    Defines the sum network in a marabou way, without the recurrent part
    i.e. we define:
        s_i b = s_i-1 f + x_i
        y = s_i f
    :param xlim: how to limit the input to the network
    :return: query to marabou that defines the sum rnn network (without recurent)
    '''
    num_params_for_cell = 8

    sum_rnn_query = MarabouCore.InputQuery()
    sum_rnn_query.setNumberOfVariables(num_params_for_cell)

    # x
    sum_rnn_query.setLowerBound(0, xlim[0])
    sum_rnn_query.setUpperBound(0, xlim[1])

    # s_i-1 f (or temp in some of my notes)
    sum_rnn_query.setLowerBound(1, 0)
    sum_rnn_query.setUpperBound(1, large)

    # s_i b
    sum_rnn_query.setLowerBound(2, -large)
    sum_rnn_query.setUpperBound(2, large)

    # s_i f
    sum_rnn_query.setLowerBound(3, 0)
    sum_rnn_query.setUpperBound(3, large)

    # z_i-1 f
    sum_rnn_query.setLowerBound(4, 0)
    sum_rnn_query.setUpperBound(4, large)

    # z_i b
    sum_rnn_query.setLowerBound(5, -large)
    sum_rnn_query.setUpperBound(5, large)

    # z_i f
    sum_rnn_query.setLowerBound(6, 0)
    sum_rnn_query.setUpperBound(6, large)

    # y
    sum_rnn_query.setLowerBound(7, -large)
    sum_rnn_query.setUpperBound(7, large)

    # s_i b = x_i * 1 + s_i-1 f * 1
    update_eq = MarabouCore.Equation()
    update_eq.addAddend(1, 0)
    update_eq.addAddend(1, 1)
    update_eq.addAddend(-1, 2)
    update_eq.setScalar(0)
    sum_rnn_query.addEquation(update_eq)

    # s_i f = ReLu(s_i b)
    MarabouCore.addReluConstraint(sum_rnn_query, 2, 3)

    # z_i b = -x_i + z_i-1 f
    update_eq = MarabouCore.Equation()
    update_eq.addAddend(-1, 0)
    update_eq.addAddend(1, 4)
    update_eq.addAddend(-1, 5)
    update_eq.setScalar(0)
    sum_rnn_query.addEquation(update_eq)

    # z_i f = ReLu(z_i b)
    MarabouCore.addReluConstraint(sum_rnn_query, 5, 6)

    # - y + skf  + zkf = 0
    output_equation = MarabouCore.Equation()
    output_equation.addAddend(1, 3)
    output_equation.addAddend(1, 6)
    output_equation.addAddend(-1, 7)
    output_equation.setScalar(0)
    sum_rnn_query.addEquation(output_equation)

    return sum_rnn_query

#</editor-fold>



def prove_invariant(xlim=(-1, 1), network_define_f=define_positive_sum_network,
                    invarinet_define_f=define_positive_sum_invariant_equations):
    '''
    proving invariant on a given rnn cell
    :param n_iterations: max number of times to run the cell
    :param input_weight: The weight for the input (before the cell)
    :param hidden_weight: The weight inside the cell
    :param xlim: limits on the input
    :param invariant_lim: what to prove (that the output of the network is smaller than some function of i)
    :return: True of the invariant holds, false otherwise
    '''

    network = network_define_f(xlim)
    base_invariant_equations, step_invariant_equations, invariant_eq = invarinet_define_f(network)

    for eq in base_invariant_equations:
        # eq.dump()
        network.addEquation(eq)

    print("Querying for induction base")
    if not marabou_solve_negate_eq(network):
        print("induction base fail")
        return False

    # TODO: Instead of creating equations again, reuse somehow (using removeEquationsByIndex, and getEquations)
    network = network_define_f(xlim)
    # base_invariant_equations, step_invariant_equations, _ = invarinet_define_f(network)
    # There is one equation we want to save in rnn_cell_query, and len(base_invariant_equations) we want to remove)
    # for i in range(len(base_invariant_equations)):
    #     rnn_cell_query.removeEquationsByIndex(1 + i)

    for eq in step_invariant_equations:
        # eq.dump()
        network.addEquation(eq)
    print("Querying for induction step")
    return marabou_solve_negate_eq(network)


def prove_property_z3(invariant_property=10, weight=1, ylim=11):
    '''
    Using z3 to probe the formula
    checking ReLu(sk * w) <= ylim[1] while sk <= sklim
    :param invariant_property: maximum value for sk
    :param weight: the weight between sk and the output
    :param ylim: max output
    :return: True if for every sk <= sklim implies that ReLu(sk * w) <= ylim
    '''

    sk = Real('sk')
    w = Real('w')
    sk_ReLU = If(sk * w >= 0, sk * w, 0)

    s = Solver()
    s.add(w == weight)
    s.add(sk_ReLU <= invariant_property)
    # we negate the condition, insted if for all sk condition we check if there exists sk not condition
    s.add(sk_ReLU * w > ylim)

    t = s.check()
    if t == sat:
        print("z3 result:", s.model())
        return False
    else:
        # print("z3 result:", t)
        return True


def prove_property_marabou(network, invariant_equations, output_equations):
    '''
    Prove property using marabou (after checking that invariant holds)
    :param network: marabou definition of the network
    :param invariant_equations: equations that the invariant promises
    :param output_equations: equations that we want to check if holds
    :return: True / False
    '''
    for eq in invariant_equations:
        # eq.dump()
        network.addEquation(eq)

    for eq in output_equations:
        # eq.dump()
        network.addEquation(eq)

    print("Querying for output")
    return marabou_solve_negate_eq(network)


def prove_using_invariant(xlim, ylim, n_iterations, network_define_f, invariant_define_f, output_define_f,
                          use_z3=False):
    '''
    Proving a property on a network using invariant's (with z3 or marabou)
    :param xlim: tuple (min, max) of the input
    :param ylim: tuple (min, max) of the output (what we want to check?)
    :param n_iterations: numebr of times to "run" the rnn cell
    :param network_define_f: pointer to function that defines the network (marabou style), gets xlim return marabou query
    :param invariant_define_f: pointer to function that defines the invariant equations, gets a network returns ([base eq, step eq, equations that hold if ivnariant holds])
    :param output_define_f: pointer to function that defines the output equations, gets, network, ylim, n_iterations return [eq to validate outputs]
    :param use_z3:
    :return:
    '''
    if not prove_invariant(xlim, network_define_f, invariant_define_f):
        print("invariant doesn't hold")
        return False

    if use_z3:
        raise NotImplementedError
        # return prove_property_z3(ylim, 1, ylim)
    else:
        network = network_define_f(xlim)
        _, _, invariant_eq = invariant_define_f(network)
        # TODO: find a better way to remove equations that were added in invariant_define_f
        network = network_define_f(xlim)
        return prove_property_marabou(network, invariant_eq, output_define_f(network, ylim, n_iterations))


if __name__ == "__main__":
    # # positive sum - sums all the positive numbers in x_1 to x_n
    # num_iterations = 500
    # pass_run = False
    # if pass_run:
    #     invariant_xlim = (-1, 1)
    # else:
    #     # invariant_xlim = (1, 1.1)
    #     invariant_xlim = (-1.1, -1)
    # y_lim = (0, num_iterations)
    # print('positive sum result:',
    #       prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_positive_sum_network,
    #                             define_positive_sum_invariant_equations,
    #                             define_positive_sum_output_equations))
    #
    # # negative sum - sums all the negative numbers in x_1 to x_n
    # num_iterations = 500
    # # pass_run = False
    # if pass_run:
    #     invariant_xlim = (-1, 1)
    # else:
    #     invariant_xlim = (-1.1, -1)
    # y_lim = (0, num_iterations)
    # print('negative sum result:',
    #       prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_negative_sum_network,
    #                             define_negative_sum_invariant_equations,
    #                             define_negative_sum_output_equations))

    pass_run = False
    num_iterations = 500
    # pass_run = False
    if pass_run:
        invariant_xlim = (0.1, 1)
        y_lim = (invariant_xlim[0] * num_iterations, invariant_xlim[1] * num_iterations)
    else:
        invariant_xlim = (0.1, 2)
        y_lim = (0, invariant_xlim[1] * num_iterations - 1)
    print('sum result max bound:',
          prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_sum_network,
                                define_sum_invariant_max_equations,
                                define_sum_max_output_equations))
    # print('sum result min bound:',
    #       prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_sum_network,
    #                             define_sum_invariant_min_equations,
    #                             define_sum_min_output_equations))
    ## last input - y == x_n
    # pass_run = False
    # num_iterations = 500
    # # pass_run = False
    # if pass_run:
    #     invariant_xlim = (-1, 1)
    #     y_lim = invariant_xlim
    # else:
    #     invariant_xlim = (-1, 2)
    #     y_lim = (-1, 0)
    # print('negative sum result:',
    #       prove_using_invariant(invariant_xlim, y_lim, num_iterations, define_last_network,
    #                             define_last_invariant_equations,
    #                             define_last_output_equations))
    # print("invariant: ", prove_invariant(n_iterations, xlim=invariant_xlim))
    # start_invariant = time.time()
    # print("invariant: ", prove_invariant(n_iterations, xlim=invariant_xlim))
    # print("sk < 10 --> y < 11:", prove_property_z3(sklim=10, weight=1, ylim=11))
    # end_invariant = time.time()
    # print("invariant + z3 took:", end_invariant - start_invariant)
    # # print("sk < 10 --> y < 4 :", prove_property_z3(sklim=10, weight=1, ylim=4))
    #
    #
