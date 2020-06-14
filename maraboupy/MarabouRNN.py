import numpy as np
from z3 import Solver, Array, BitVec, BitVecSort, RealSort, ForAll, sat, BV2Int, BitVecVal

from maraboupy import MarabouCore

large = 50000.0
small = 10 ** -2
TOLERANCE_VALUE = 0.01
ALPHA_IMPROVE_EACH_ITERATION = 10


def marabou_solve_negate_eq(query, debug=False):
    '''
    Run marabou solver
    :param query: query to execute
    :param debug: if True printing all of the query equations
    :return: True if UNSAT (no valid assignment), False otherwise
    '''
    # if debug:
    #     for eq in query.getEquations():
    #         eq.dump()

    vars1, stats1 = MarabouCore.solve(query, "", 0)
    if len(vars1) > 0:
        print("SAT")
        print(vars1)
        return False
    else:
        print("UNSAT")
        return True


def negate_equation(eq):
    '''
    negates the equation
    :param eq: equation
    :return: new equation which is exactly (not eq)
    '''
    not_eq = MarabouCore.Equation(eq)
    if eq.getType() == MarabouCore.Equation.GE:
        not_eq.setType(MarabouCore.Equation.LE)
        not_eq.setScalar(eq.getScalar() - small)
    elif eq.getType() == MarabouCore.Equation.LE:
        not_eq.setType(MarabouCore.Equation.GE)
        not_eq.setScalar(eq.getScalar() + small)
    elif eq.setType(MarabouCore.Equation.EQ):
        raise NotImplementedError("can't negate equal equations")
    else:
        raise NotImplementedError("got {} type which is not implemented".format(eq.getType()))
    return not_eq


def add_rnn_cell(query, input_weights, hidden_weight, num_iterations, bias=0, print_debug=False):
    '''
    Create rnn cell --> add 4 parameters to the query and the equations that describe the cell
    The added parameters are (same order): i, s_i-1 f, s_i b, s_i f
    :param query: the network so far (will add to this)
    :param input_weights: list of tuples, each tuple (variable_idx, weight)
    :param hidden_weight: the weight inside the cell
    :param num_iterations: Number of iterations the cell runs
    :return: the index of the last parameter (which is the output of the cell)
    '''

    last_idx = query.getNumberOfVariables()
    query.setNumberOfVariables(last_idx + 4)  # i, s_i-1 f, s_i b, s_i f

    # i
    # TODO: when doing this we make the number of iterations to be n_iterations + 1
    query.setLowerBound(last_idx, 0)
    query.setUpperBound(last_idx, num_iterations)

    # s_i-1 f
    query.setLowerBound(last_idx + 1, 0)
    query.setUpperBound(last_idx + 1, large)

    # s_i b
    query.setLowerBound(last_idx + 2, -large)
    query.setUpperBound(last_idx + 2, large)

    # s_i f
    query.setLowerBound(last_idx + 3, 0)
    query.setUpperBound(last_idx + 3, large)

    # s_i f = ReLu(s_i b)
    MarabouCore.addReluConstraint(query, last_idx + 2, last_idx + 3)

    # s_i-1 f >= i * \sum (x_j_min * w_j)
    # prev_min_eq = MarabouCore.Equation(MarabouCore.Equation.LE)
    # prev_min_eq.addAddend(1, last_idx + 1)
    # prev_min_eq.addAddend(1, last_idx + 1)

    # s_i b = x_j * w_j for all j connected + s_i-1 f * hidden_weight
    update_eq = MarabouCore.Equation()
    for var_idx, weight in input_weights:
        update_eq.addAddend(weight, var_idx)
    update_eq.addAddend(hidden_weight, last_idx + 1)
    update_eq.addAddend(-1, last_idx + 2)
    update_eq.setScalar(-bias)
    # if print_debug:
    #     update_eq.dump()
    query.addEquation(update_eq)

    return last_idx + 3


def add_loop_indices_equations(network, loop_indices):
    '''
    Adds to the network equations that make all loop variabels to be equal
    :param network: marabou quert that the equations will be appended
    :param loop_indices: variables that needs to be equal
    :return: None
    '''
    # Make sure all the iterators are in the same iteration, we create every equation twice
    step_loop_eq = []
    for idx in loop_indices:
        for idx2 in loop_indices:
            if idx < idx2:
                temp_eq = MarabouCore.Equation()
                temp_eq.addAddend(1, idx)
                temp_eq.addAddend(-1, idx2)
                # step_loop_eq.append(temp_eq)
                network.addEquation(temp_eq)


def create_invariant_equations(loop_indices, invariant_eq):
    '''
    create the equations needed to prove using induction from the invariant_eq
    :param loop_indices: List of loop variables (i's), which is the first variable for an RNN cell
    :param invariant_eq: the invariant we want to prove
    :return: [base equations], [step equations]
    '''

    def create_induction_hypothesis_from_invariant_eq():
        '''
        for example our invariant is that s_i f <= i, the induction hypothesis will be s_i-1 f <= i-1
        :return: the induction hypothesis
        '''
        scalar_diff = 0
        hypothesis_eq = []

        cur_temp_eq = MarabouCore.Equation(invariant_eq.getType())
        for addend in invariant_eq.getAddends():
            # if for example we have s_i f - 2*i <= 0 we want s_i-1 f - 2*(i-1) <= 0 <--> s_i-1 f -2i <= -2
            if addend.getVariable() in loop_indices:
                scalar_diff = addend.getCoefficient()
            # here we change s_i f to s_i-1 f
            if addend.getVariable() in rnn_output_indices:
                cur_temp_eq.addAddend(addend.getCoefficient(), addend.getVariable() - 2)
            else:
                cur_temp_eq.addAddend(addend.getCoefficient(), addend.getVariable())
        cur_temp_eq.setScalar(invariant_eq.getScalar() + scalar_diff)
        hypothesis_eq.append(cur_temp_eq)
        return hypothesis_eq

    rnn_input_indices = [idx + 1 for idx in loop_indices]
    rnn_output_indices = [idx + 3 for idx in loop_indices]

    # equations for induction step
    if isinstance(invariant_eq, list):
        if len(invariant_eq) == 1:
            invariant_eq = invariant_eq[0]
        else:
            raise Exception

    induction_step = negate_equation(invariant_eq)

    # equations for induction base

    # make sure i == 0 (for induction base)
    loop_equations = []
    for i in loop_indices:
        loop_eq = MarabouCore.Equation()
        loop_eq.addAddend(1, i)
        loop_eq.setScalar(0)
        loop_equations.append(loop_eq)

    # s_i-1 f == 0
    zero_rnn_hidden = []
    for idx in rnn_input_indices:
        base_hypothesis = MarabouCore.Equation()
        base_hypothesis.addAddend(1, idx)
        base_hypothesis.setScalar(0)
        zero_rnn_hidden.append(base_hypothesis)

    step_loop_eq = MarabouCore.Equation(MarabouCore.Equation.GE)
    step_loop_eq.addAddend(1, loop_indices[0])
    step_loop_eq.setScalar(1)
    # step_loop_eq.append(step_loop_eq_more_1)

    induction_base_equations = [induction_step] + loop_equations + zero_rnn_hidden

    induction_hypothesis = create_induction_hypothesis_from_invariant_eq()
    induction_step_equations = [induction_step] + [step_loop_eq]
    # induction_step_equations = induction_step + step_loop_eq

    return induction_base_equations, induction_step_equations, induction_hypothesis


def prove_adversarial_property_marabou(a_pace, b_pace, min_a, max_b, n_iterations):
    min_a_n = min_a + a_pace * n_iterations
    max_b_n = max_b + b_pace * n_iterations
    return not min_a_n < max_b_n


def prove_adversarial_property_z3(a_pace, b_pace, min_a, max_b, n_iterations):
    '''
    Using z3 to probe the formula
    checking ReLu(sk * w) <= ylim[1] while sk <= sklim
    :param invariant_property: maximum value for sk
    :param weight: the weight between sk and the output
    :param ylim: max output
    :return: True if for every sk <= sklim implies that ReLu(sk * w) <= ylim
    '''
    from math import log2, ceil
    num_bytes = ceil(log2(n_iterations)) + 1
    print('n_iterations', n_iterations, '\nnum_bytes', num_bytes)
    assert n_iterations <= 2 ** num_bytes  # if the bit vec is 32 z3 takes to long
    a_invariants = Array('a_invariants', BitVecSort(num_bytes), RealSort())
    b_invariants = Array('b_invariants', BitVecSort(num_bytes), RealSort())
    i = BitVec('i', num_bytes)
    n = BitVec('n', num_bytes)

    s = Solver()
    s.add(a_invariants[0] == min_a)
    s.add(b_invariants[0] == max_b)
    s.add(n == BitVecVal(n_iterations, num_bytes))

    # The invariant
    s.add(ForAll(i, a_invariants[i] >= a_invariants[0] + BV2Int(i) * a_pace))
    s.add(ForAll(i, b_invariants[i] <= b_invariants[0] + BV2Int(i) * b_pace))
    # s.add(ForAll(i, a_invariants[i] >= a_invariants[0] + BV2Int(i * BitVecVal(a_pace, num_bytes))))
    # s.add(ForAll(i, b_invariants[i] <= b_invariants[0] + BV2Int(i * BitVecVal(b_pace, num_bytes))))

    # NOT the property to prove
    s.add(a_invariants[n] < b_invariants[n])

    t = s.check()
    if t == sat:
        print("z3 result:", s.model())
        return False
    else:
        print('proved adversarial property using z3')
        return True


def prove_property_marabou(network, invariant_equations, output_equations, iterators_idx, n_iterations):
    '''
    Prove property using marabou (after checking that invariant holds)
    :param network: marabou definition of the network
    :param invariant_equations: equations that the invariant promises
    :param output_equations: equations that we want to check if holds
    :return: True if the property holds, False otherwise
    '''
    added_equations = []
    if iterators_idx:
        for idx in iterators_idx:
            iterator_eq = MarabouCore.Equation()
            iterator_eq.addAddend(1, idx)
            iterator_eq.setScalar(n_iterations)
            added_equations.append(iterator_eq)
            network.addEquation(iterator_eq)

    not_output = []
    for eq in output_equations:
        not_output.append(negate_equation(MarabouCore.Equation(eq)))

    if invariant_equations:
        for eq in invariant_equations:
            added_equations.append(eq)
            network.addEquation(eq)

    for eq in not_output:
        added_equations.append(eq)
        network.addEquation(eq)

    print("prove property on marabou:")
    network.dump()
    ret_val = marabou_solve_negate_eq(network, True)

    for eq in added_equations:
        network.removeEquation(eq)
    return ret_val


def simplify_network_using_invariants2(network, rnn_cells):
    pass


def simplify_network_using_invariants(network_define_f, xlim, ylim, n_iterations):
    network, rnn_start_idxs, invariant_equation, *_ = network_define_f(xlim, ylim, n_iterations)

    for idx in rnn_start_idxs:
        for idx2 in rnn_start_idxs:
            if idx != idx2:
                temp_eq = MarabouCore.Equation()
                temp_eq.addAddend(1, idx)
                temp_eq.addAddend(-1, idx2)
                network.addEquation(temp_eq)

    if not isinstance(invariant_equation, list):
        invariant_equation = [invariant_equation]

    for i in range(len(invariant_equation)):
        if not prove_invariant2(network, [rnn_start_idxs[i]], [invariant_equation[i]]):
            print("Fail on invariant: ", i)
            return False
        else:
            # Add the invariant hypothesis for the next proving
            network.addEquation(invariant_equation[i])

    return True


def prove_invariant_multi_base(network, rnn_start_idxs, invariant_equations):
    '''
    Prove invariants where we need to assume multiple assumptions and conclude from them.
    For each of the invariant_equations creating 3 sets: base_equations, hyptosis_equations, step_equations
    First proving on each of the base equations seperatly, Then assuming all the hyptosis equations and proving
    on the step_equations set by set
    :param network:
    :param rnn_start_idxs:
    :param invariant_equations:
    :return:
    '''
    base_eq = []
    step_eq = []  # this needs to be a list of lists, each time we work on all equations of a list
    hypothesis_eq = []
    for i in range(len(invariant_equations)):
        cur_base_eq, cur_step_eq,  cur_hypothesis_eq = create_invariant_equations(rnn_start_idxs, invariant_equations)
        base_eq.append(cur_base_eq)
        step_eq.append(cur_step_eq)
        hypothesis_eq += cur_hypothesis_eq

    # first prove base case for all equations
    for ls_eq in base_eq:
        for eq in ls_eq:
            network.addEquation(eq)
        marabou_result = marabou_solve_negate_eq(network)
        for eq in ls_eq:
            network.removeEquation(eq)

        if not marabou_result:
            print("induction base fail, on invariant:", i)
            return False

    # add all hypothesis equations
    for eq in hypothesis_eq:
        print("hypothesis_eq")
        eq.dump()
        network.addEquation(eq)

    for steq_eq_ls in step_eq:
        for eq in steq_eq_ls:
            # eq.dump()
            network.addEquation(eq)

        print("Querying for induction step")
        network.dump()

        marabou_result = marabou_solve_negate_eq(network)
        for eq in steq_eq_ls:
            network.removeEquation(eq)

        if not marabou_result:
            for eq in hypothesis_eq:
                network.removeEquation(eq)
            print("induction step fail, on invariant:", i)
            return False

    for eq in hypothesis_eq:
        network.removeEquation(eq)
    return True


def prove_invariant2(network, rnn_start_idxs, invariant_equations):
    '''
    proving invariant network using induction (proving for the first iteration, and concluding that after iteration k
       the property holds assuming k-1 holds)
    Not changing the network, i.e. every equation we add we also remove
    :param network: description of the NN in marabou style
    :param invariant_equations: List of Marabou equations that describe the current invariant
    :return: True if the invariant holds, false otherwise
    '''

    for i in range(len(invariant_equations)):
        base_equations, step_equations, hypothesis_eq = create_invariant_equations(rnn_start_idxs, [invariant_equations[i]])

        step_equations += hypothesis_eq
        for eq in base_equations:
            # eq.dump()
            network.addEquation(eq)

        print("Querying for induction base")
        # network.dump()

        marabou_result = marabou_solve_negate_eq(network)
        for eq in base_equations:
            network.removeEquation(eq)

        if not marabou_result:
            print("induction base fail, on invariant:", i)
            return False

        for eq in base_equations:
            network.removeEquation(eq)

        for eq in step_equations:
            # eq.dump()
            network.addEquation(eq)

        print("Querying for induction step")
        # network.dump()

        marabou_result = marabou_solve_negate_eq(network)
        for eq in step_equations:
            network.removeEquation(eq)

        if not marabou_result:
            # network.dump()
            print("induction step fail, on invariant:", i)
            return False

    return True


def find_stronger_invariant(network, max_alphas, min_alphas, i, rnn_output_idxs, rnn_start_idxs,
                            initial_values, eq_type=MarabouCore.Equation.GE):
    counter = 0
    cur_alpha = (max_alphas[i] + min_alphas[i]) / 2
    proven_invariant_equation = None
    while max_alphas[i] - min_alphas[i] > TOLERANCE_VALUE and counter < ALPHA_IMPROVE_EACH_ITERATION:
        invariant_equation = MarabouCore.Equation(eq_type)
        invariant_equation.addAddend(1, rnn_output_idxs[i])  # b_i
        if eq_type == MarabouCore.Equation.LE:
            ge_better = -1
        else:
            ge_better = 1
        invariant_equation.addAddend(cur_alpha * ge_better, rnn_start_idxs[i])  # i
        invariant_equation.setScalar(initial_values[i])
        prove_inv_res = prove_invariant2(network, rnn_start_idxs, [invariant_equation])
        # prove_inv_res = prove_invariant2(network, rnn_start_idxs, [invariant_equation])
        if prove_inv_res:
            print("For alpha_{} {} invariant holds".format(i, cur_alpha))
            proven_invariant_equation = invariant_equation
            max_alphas[i] = cur_alpha
            counter += 1
            # return min_alphas, max_alphas, cur_alpha, invariant_equation
        else:
            print("For alpha_{} {} invariant does not hold".format(i, cur_alpha))
            # Invariant does not hold
            min_alphas[i] = cur_alpha
        cur_alpha = (max_alphas[i] + min_alphas[i]) / 2  # weaker invariant

        # cur_alpha = temp
    cur_alpha = None if proven_invariant_equation is None else cur_alpha
    return min_alphas, max_alphas, cur_alpha, proven_invariant_equation


def find_stronger_invariant_multidim(network, max_alphas, min_alphas, i, rnn_output_idxs, rnn_start_idxs,
                            initial_values, eq_type=MarabouCore.Equation.GE):
    counter = 0
    cur_alpha = (max_alphas[i] + min_alphas[i]) / 2
    proven_invariant_equation = None
    while max_alphas[i] - min_alphas[i] > TOLERANCE_VALUE and counter < ALPHA_IMPROVE_EACH_ITERATION:
        invariant_equation = MarabouCore.Equation(eq_type)
        invariant_equation.addAddend(1, rnn_output_idxs[i])  # b_i
        if eq_type == MarabouCore.Equation.LE:
            ge_better = -1
        else:
            ge_better = 1
        invariant_equation.addAddend(cur_alpha * ge_better, rnn_start_idxs[i])  # i
        invariant_equation.setScalar(initial_values[i])
        prove_inv_res = prove_invariant2(network, rnn_start_idxs, [invariant_equation])
        # prove_inv_res = prove_invariant2(network, rnn_start_idxs, [invariant_equation])
        if prove_inv_res:
            print("For alpha_{} {} invariant holds".format(i, cur_alpha))
            proven_invariant_equation = invariant_equation
            max_alphas[i] = cur_alpha
            counter += 1
            # return min_alphas, max_alphas, cur_alpha, invariant_equation
        else:
            print("For alpha_{} {} invariant does not hold".format(i, cur_alpha))
            # Invariant does not hold
            min_alphas[i] = cur_alpha
        cur_alpha = (max_alphas[i] + min_alphas[i]) / 2  # weaker invariant

        # cur_alpha = temp
    cur_alpha = None if proven_invariant_equation is None else cur_alpha
    return min_alphas, max_alphas, cur_alpha, proven_invariant_equation


def improve_invariant_multidim(network, rnn_output_idxs, rnn_start_idxs, initial_values, rnn_invariant_type,
                      invariant_that_hold, max_alphas, min_alphas, i, dependent_cells, prev_alpha):

    still_improve = True
    new_alpha = prev_alpha
    if invariant_that_hold:
        network.removeEquation(invariant_that_hold)

    min_alphas, max_alphas, temp_alpha, cur_inv_eq = find_stronger_invariant_multidim(network, max_alphas,
                                                                             min_alphas,
                                                                             i,
                                                                             rnn_output_idxs,
                                                                             rnn_start_idxs,
                                                                             initial_values,
                                                                             rnn_invariant_type[i])

    if temp_alpha is not None:
        new_alpha = temp_alpha
        invariant_that_hold = cur_inv_eq
        # Found a better invariant need to change the search space for all the rest
        if dependent_cells:
            print("found invariant for: {}, zeroing: {}".format(i, dependent_cells))
            for j in dependent_cells:
                max_alphas[j] = large
                min_alphas[j] = -large
                still_improve = True

    if max_alphas[i] - min_alphas[i] <= TOLERANCE_VALUE:
        still_improve = False

    # This invariant hold, all other rnn cells can use this fact
    network.addEquation(invariant_that_hold)
    return still_improve, new_alpha, invariant_that_hold


def improve_invariant(network, rnn_output_idxs, rnn_start_idxs, initial_values, rnn_invariant_type,
                      invariant_that_hold, max_alphas, min_alphas, i, dependent_cells, prev_alpha):
    still_improve = True
    new_alpha = prev_alpha
    if invariant_that_hold:
        network.removeEquation(invariant_that_hold)

    min_alphas, max_alphas, temp_alpha, cur_inv_eq = find_stronger_invariant(network, max_alphas,
                                                                             min_alphas,
                                                                             i,
                                                                             rnn_output_idxs,
                                                                             rnn_start_idxs,
                                                                             initial_values,
                                                                             rnn_invariant_type[i])

    if temp_alpha is not None:
        new_alpha = temp_alpha
        invariant_that_hold = cur_inv_eq
        # Found a better invariant need to change the search space for all the rest
        if dependent_cells:
            print("found invariant for: {}, zeroing: {}".format(i, dependent_cells))
            for j in dependent_cells:
                max_alphas[j] = large
                min_alphas[j] = -large
                still_improve = True

    if max_alphas[i] - min_alphas[i] <= TOLERANCE_VALUE:
        still_improve = False

    # This invariant hold, all other rnn cells can use this fact
    network.addEquation(invariant_that_hold)
    return still_improve, new_alpha, invariant_that_hold


def find_invariant_adversarial(network, rnn_start_idxs, rnn_invariant_type, initial_values, n_iterations,
                               min_alphas=None,
                               max_alphas=None, rnn_dependent=None):
    '''
    Function to automatically find invariants that hold and prove the property
    The order of the rnn indices matter (!), we try to prove invariants on them sequentially,
    i.e. if rnn_x is dependent on the output of rnn_y then index(rnn_x) > index(rnn_y)
    :param network: Description of the network in Marabou style
    :param rnn_start_idxs: list of indcies with the iterator variable for each rnn cell
    :param rnn_invariant_type: List of MarabouCore.Equation.GE/LE, for each RNN cell
    :param initial_values: (min_a, max_b)
    :param n_iterations: for how long we will run the network (how many inputs will there be)
    :param min_alphas:
    :param max_alphas:
    :param rnn_dependent: list of lists (or none), for each cell which rnn are dependent on him. we need this to
            recompute the search space after finiding a better invariant
    :return:
    '''

    assert len(rnn_start_idxs) == len(rnn_invariant_type)
    for t in rnn_invariant_type:
        assert t == MarabouCore.Equation.GE or t == MarabouCore.Equation.LE
    if not rnn_dependent:
        rnn_dependent = [None] * len(rnn_start_idxs)
    assert len(rnn_dependent) == len(rnn_invariant_type)

    rnn_output_idxs = [i + 3 for i in rnn_start_idxs]

    add_loop_indices_equations(network, rnn_start_idxs)

    invariant_equation = None

    if initial_values:
        initial_diff = initial_values[-2] - initial_values[-1]
        assert initial_diff >= 0

    # TODO: Find suitable range for the invariant to be in
    if min_alphas is None:
        min_alphas = [-large] * len(rnn_start_idxs)  # min alpha that we try but invariant does not hold
    if max_alphas is None:
        max_alphas = [large] * len(rnn_start_idxs)  # max alpha that we try but property does not hold

    # For A small alpha yields stronger invariant, while B is the opposite
    alphas = []
    for i, inv_type in enumerate(rnn_invariant_type):
        if inv_type == MarabouCore.Equation.GE:
            alphas.append(min_alphas[i])
        else:
            alphas.append(max_alphas[i])

    still_improve = [True] * len(rnn_start_idxs)

    # Keep track on the invariants we now that hold for each cell
    invariant_that_hold = [None] * len(rnn_start_idxs)
    while any(still_improve):

        for i in range(len(rnn_start_idxs)):
            if still_improve[i]:
                still_improve[i], alphas[i], invariant_that_hold[i] = improve_invariant(network, rnn_output_idxs,
                                                                                        rnn_start_idxs, initial_values,
                                                                                        rnn_invariant_type,
                                                                                        invariant_that_hold[i],
                                                                                        max_alphas, min_alphas, i,
                                                                                        rnn_dependent[i], alphas[i])

        # TODO: Need to change this, no sense to take the last two alphas, probably need to prove an invariant on A and B also and not only the RNN's
        if prove_adversarial_property_marabou(-alphas[-2], alphas[-1], initial_values[-2], initial_values[-1],
                                              n_iterations):
            # if prove_adversarial_property_z3(-alphas[-2], alphas[-1], initial_values[-2], initial_values[-1], n_iterations):
            print("Invariant and property holds. invariants:\n\t" + "\n\t".join(
                ["alpha_{}: {}".format(i, a) for i, a in enumerate(alphas)]))
            return True
        else:
            print("Property does not fold for alphas:\n\t" + "\n\t".join(
                ["{}: {}".format(i, a) for i, a in enumerate(alphas)]))

    network.dump()
    print("Finish trying to find sutiable invariant, the last invariants we found are\n\t" + "\n\t".join(
        ["alpha_{}: {}".format(i, a) for i, a in enumerate(alphas)]))

    print("last search area\n\t" + "\n\t".join(
        ["{}: {} TO {}".format(i, min_a, max_a) for i, (min_a, max_a) in enumerate(zip(min_alphas, max_alphas))]))
    return False


def find_invariant_marabou(network, rnn_start_idxs, rnn_invariant_type, initial_values, n_iterations,
                           property_equations, min_alphas=None, max_alphas=None, rnn_dependent=None):
    '''
    Function to automatically find invariants that hold and prove the property
    The order of the rnn indices matter (!), we try to prove invariants on them sequentially,
    i.e. if rnn_x is dependent on the output of rnn_y then index(rnn_x) > index(rnn_y)
    :param network: Description of the network in Marabou style
    :param rnn_start_idxs: list of indicies of the iterator variable for each rnn cell
    :param rnn_invariant_type: List of MarabouCore.Equation.GE/LE, for each RNN cell
    :param initial_values: (min_a, max_b)
    :param n_iterations: for how long we will run the network (how many inputs will there be)
    :param min_alphas:
    :param max_alphas:
    :param rnn_dependent: list of lists (or none), for each cell which rnn are dependent on him. we need this to
            recompute the search space after finiding a better invariant
    :return:
    '''

    assert len(rnn_start_idxs) == len(rnn_invariant_type)
    for t in rnn_invariant_type:
        assert t == MarabouCore.Equation.GE or t == MarabouCore.Equation.LE
    if not rnn_dependent:
        rnn_dependent = [None] * len(rnn_start_idxs)
    assert len(rnn_dependent) == len(rnn_invariant_type)

    rnn_output_idxs = [i + 3 for i in rnn_start_idxs]
    invariant_equation = None
    assert invariant_equation is None

    # if initial_values and len(initial_values) >= 2:
    #     initial_diff = initial_values[-2] - initial_values[-1]
    #     assert initial_diff >= 0

    # TODO: Find suitable range for the invariant to be in
    if min_alphas is None:
        min_alphas = [-large] * len(rnn_start_idxs)  # min alpha that we try but invariant does not hold
    if max_alphas is None:
        max_alphas = [large] * len(rnn_start_idxs)  # max alpha that we try but property does not hold

    # For A small alpha yields stronger invariant, while B is the opposite
    alphas = []
    for i, inv_type in enumerate(rnn_invariant_type):
        if inv_type == MarabouCore.Equation.GE:
            alphas.append(min_alphas[i])
        else:
            alphas.append(max_alphas[i])

    still_improve = [True] * len(rnn_start_idxs)

    # Keep track on the invariants we now that hold for each cell
    invariant_that_hold = [None] * len(rnn_start_idxs)
    while any(still_improve):
        network.dump()

        for i in range(len(rnn_start_idxs)):
            if still_improve[i]:
                still_improve[i], alphas[i], invariant_that_hold[i] = improve_invariant(network, rnn_output_idxs,
                                                                                        rnn_start_idxs, initial_values,
                                                                                        rnn_invariant_type,
                                                                                        invariant_that_hold[i],
                                                                                        max_alphas, min_alphas, i,
                                                                                        rnn_dependent[i], alphas[i])

        if prove_property_marabou(network, None, property_equations, None, n_iterations):
            print("Invariant and property holds. invariants:\n\t" + "\n\t".join(
                ["alpha_{}: {}".format(i, a) for i, a in enumerate(alphas)]))
            return True
        else:
            print("Property does not fold for alphas:\n\t" + "\n\t".join(
                ["{}: {}".format(i, a) for i, a in enumerate(alphas)]))


    network.dump()
    print("Finish trying to find sutiable invariant, the last invariants we found are\n\t" + "\n\t".join(
        ["alpha_{}: {}".format(i, a) for i, a in enumerate(alphas)]))

    print("last search area\n\t" + "\n\t".join(
        ["{}: {} TO {}".format(i, min_a, max_a) for i, (min_a, max_a) in enumerate(zip(min_alphas, max_alphas))]))
    return False


def find_invariant_marabou_multidim(network, rnn_start_idxs, rnn_invariant_type, initial_values, n_iterations, property_equations,
                                    rnn_dependent=None):
    '''
    Function to automatically find invariants that hold and prove the property
    The order of the rnn indices matter (!), we try to prove invariants on them sequentially,
    i.e. if rnn_x is dependent on the output of rnn_y then index(rnn_x) > index(rnn_y)
    :param network: Description of the network in Marabou style
    :param rnn_start_idxs: list of lists, each list if a set of rnns we need to prove together, then each cell is index
     with the iterator variable for each rnn cell
    :param rnn_invariant_type: List of MarabouCore.Equation.GE/LE, for each RNN cell
    :param n_iterations: for how long we will run the network (how many inputs will there be)
    :param rnn_dependent: list of lists (or none), for each cell which rnn are dependent on him. we need this to
            recompute the search space after finding a better invariant
    :return:
    '''

    assert len(rnn_start_idxs) == len(rnn_invariant_type)
    for t in rnn_invariant_type:
        assert t == MarabouCore.Equation.GE or t == MarabouCore.Equation.LE
    if not rnn_dependent:
        rnn_dependent = [None] * len(rnn_start_idxs)
    assert len(rnn_dependent) == len(rnn_invariant_type)

    rnn_output_idxs = [i + 3 for i in rnn_start_idxs]
    invariant_equation = None
    assert invariant_equation is None

    # TODO: Find suitable range for the invariant to be in
    min_alphas = [-large] * len(rnn_start_idxs)  # min alpha that we try but invariant does not hold
    max_alphas = [large] * len(rnn_start_idxs)  # max alpha that we try but property does not hold

    # For A small alpha yields stronger invariant, while B is the opposite
    alphas = []
    for i, inv_type in enumerate(rnn_invariant_type):
        if inv_type == MarabouCore.Equation.GE:
            alphas.append(min_alphas[i])
        else:
            alphas.append(max_alphas[i])

    still_improve = [True] * len(rnn_start_idxs)

    # Keep track on the invariants we now that hold for each cell
    invariant_that_hold = [None] * len(rnn_start_idxs)
    while any(still_improve):

        for i in range(len(rnn_start_idxs)):
            if still_improve[i]:
                still_improve[i], alphas[i], invariant_that_hold[i] = improve_invariant_multidim(network, rnn_output_idxs,
                                                                                        rnn_start_idxs, initial_values,
                                                                                        rnn_invariant_type,
                                                                                        invariant_that_hold[i],
                                                                                        max_alphas, min_alphas, i,
                                                                                        rnn_dependent[i], alphas[i])


        if prove_property_marabou(network, None, property_equations, None, n_iterations):
            print("Invariant and property holds. invariants:\n\t" + "\n\t".join(
                ["alpha_{}: {}".format(i, a) for i, a in enumerate(alphas)]))
            return True
        else:
            print("Property does not fold for alphas:\n\t" + "\n\t".join(
                ["{}: {}".format(i, a) for i, a in enumerate(alphas)]))

    network.dump()
    print("Finish trying to find sutiable invariant, the last invariants we found are\n\t" + "\n\t".join(
        ["alpha_{}: {}".format(i, a) for i, a in enumerate(alphas)]))

    print("last search area\n\t" + "\n\t".join(
        ["{}: {} TO {}".format(i, min_a, max_a) for i, (min_a, max_a) in enumerate(zip(min_alphas, max_alphas))]))
    return False


def prove_using_invariant(xlim, ylim, n_iterations, network_define_f, use_z3=False):
    '''
    Proving a property on a network using invariant's (with z3 or marabou)
    :param xlim: tuple (min, max) of the input
    :param ylim: tuple (min, max) of the output (what we want to check?)
    :param n_iterations: number of times to "run" the rnn cell
    :param network_define_f: function that returns the marabou network, invariant, output property
    :param use_z3:
    :return: True if the invariant holds and we can conclude the property from it, False otherwise
    '''
    if not simplify_network_using_invariants(network_define_f, xlim, ylim, n_iterations):
        print("invariant doesn't hold")
        return False
    if use_z3:
        raise NotImplementedError
        # return prove_property_z3(ylim, 1, ylim)
    else:
        network, iterators_idx, invariant_equation, output_eq, *_ = network_define_f(xlim, ylim, n_iterations)
        # inv_eq = MarabouCore.Equation(MarabouCore.Equation.GE)

        if not isinstance(invariant_equation, list):
            invariant_equation = [invariant_equation]
        return prove_property_marabou(network, invariant_equation, output_eq, iterators_idx, n_iterations)


def prove_adversarial_using_invariant(xlim, n_iterations, network_define_f):
    '''
    Proving a property on a network using invariant's (with z3 or marabou)
    :param xlim: tuple (min, max) of the input
    :param ylim: tuple (min, max) of the output (what we want to check?)
    :param n_iterations: number of times to "run" the rnn cell
    :param network_define_f: function that returns the marabou network, invariant, output property
    :param use_z3:
    :return: True if the invariant holds and we can conclude the property from it, False otherwise
    '''

    # Use partial define because in the meantime network_define_f gets also ylim which in this case we don't need
    # The partial allows us to use a generic prove_invariant for both cases
    partial_define = lambda xlim, ylim, n_iterations: network_define_f(xlim, n_iterations)

    if not simplify_network_using_invariants(partial_define, xlim, None, n_iterations):
        print("invariant doesn't hold")
        return False

    _, _, _, (min_a, max_b), (a_pace, b_pace), *_ = network_define_f(xlim, n_iterations)
    return prove_adversarial_property_z3(a_pace, b_pace, min_a, max_b, n_iterations)
