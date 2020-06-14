from maraboupy import MarabouCore
from maraboupy.MarabouRNN import *


def relu(num):
    return max(0, num)


def adversarial_robustness_sum_results(x):
    '''
    Get a list of inputs and calculate A,B according to the robustness network
    :param x:
    :return:
    '''

    s_i_1_f = 0
    z_i_1_f = 0

    for num in x:
        s_i_f = relu(2 * num + 1 * s_i_1_f)
        z_i_f = relu(1 * num + 1 * z_i_1_f)
        s_i_1_f = s_i_f
        z_i_1_f = z_i_f

    A = 2 * s_i_f  # + z_i_f
    B = 2 * z_i_f  # + s_i_f

    return A, B


def z3_adversarial_unfold(n=10, to_pass=True):
    from z3 import Reals, Solver, sat
    s = Solver()
    A0, B0 = Reals('A0 B0')
    s.add(A0 == 0)
    # if to_pass:
    s.add(B0 == n)
    #
    # s.add(B0 == n+1)

    A_prev = A0
    B_prev = B0
    for i in range(1, n + 1):
        A_temp, B_temp = Reals('A{} B{}'.format(i, i))
        # This is the invariant
        if to_pass:
            s.add(A_temp == A_prev + 1)
        else:
            s.add(A_temp == A_prev + 0.9)
        s.add(B_temp == B_prev)
        A_prev = A_temp
        B_prev = B_temp

    s.add(A_temp < B_temp)

    # print(s)
    t = s.check()
    if t == sat:
        # print("z3 result:", s.model())
        return False
    else:
        # print("z3 result:", t)
        return True


def check_adversarial_robustness_z3():
    from z3 import Reals, Int, Solver, If, And
    sk, sk_1, zk, zk_1 = Reals('sk sk_1 zk zk_1')
    i = Int('i')

    s = Solver()
    s.add(And(i >= 0, i <= 20, sk_1 >= 0, sk >= 0, zk >= 0, zk_1 >= 0))
    A = If(sk * 1 >= 0, sk * 1, 0)
    B = If(zk * 1 >= 0, zk * 1, 0)

    s.add(If(i == 0,
             And(sk >= 0, sk <= 3, zk >= 10, zk <= 21, sk_1 == 0, zk_1 == 0),
             sk - zk >= sk_1 - zk_1 + 21 / i))

    s.add(And(A < B, i == 20))
    # # we negate the condition, instead if for all sk condition we check if there exists sk not condition
    # s.add(sk_ReLU * w > ylim)

    t = s.check()
    if t == sat:
        print("z3 result:", s.model())
        return False
    else:
        # print("z3 result:", t)
        return True


def define_adversarial_robustness_two_input_nodes_step_fail(xlim, n_iterations):
    '''
    Define an adversarial robustness examples, where it will not be possible to find an invariant that will work
    0 <= x_0 <= 1
    1 <= x_1 <= 2
    s_i = 1 * x_0 + 5 * x_1 + 1 * s_(i-1)
    z_i = 2 * x_0 + 1 * x_1 + 100 * z_(i-1)
    A = s_i
    B = z_i
    therefore:
        5 <= A <= 6
        1 <= B <= 4
    prove that after n_iterations A >= B
    :param xlim: array of tuples, each array cell as an input (x_0, x_1 etc..) tuple is (min_value, max_value)
    :param n_iterations: number of iterations
    :return: network, [s_cell_iterator, z_cell_iterator], [a_invariant_equation, b_invariant_equation],\
           (min_a, max_b), [a_base_equation, b_base_equation], (-alpha, alpha)
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
    z_hidden_w = 100
    x0_s_w = 1
    x1_s_w = 5
    x0_z_w = 2
    x1_z_w = 1

    # s_i_f = relu(2 * x1 + 1 * x2 + 1.5*s_i-1_f)
    s_cell_iterator = network.getNumberOfVariables()  # i
    s_i_f_idx = add_rnn_cell(network, [(0, x0_s_w), (1, x1_s_w)], s_hidden_w, n_iterations, print_debug=True)
    s_i_1_f_idx = s_i_f_idx - 2
    # z_i_f = relu(1 * x1 + 10 * x2 + z_i-1_f)
    z_cell_iterator = network.getNumberOfVariables()
    z_i_f_idx = add_rnn_cell(network, [(0, x0_z_w), (1, x1_z_w)], z_hidden_w, n_iterations, print_debug=True)
    z_i_1_f_idx = z_i_f_idx -

    a_idx = z_i_f_idx + 1
    b_idx = a_idx + 1

    a_w = 1
    b_w = 1

    network.setNumberOfVariables(b_idx + 1)

    # A
    network.setLowerBound(a_idx, -large)  # A
    network.setUpperBound(a_idx, large)

    # B
    network.setLowerBound(b_idx, -large)  # B
    network.setUpperBound(b_idx, large)

    # # A = skf <--> A - skf = 0
    a_output_eq = MarabouCore.Equation()
    a_output_eq.addAddend(1, a_idx)
    a_output_eq.addAddend(-a_w, s_i_f_idx)
    a_output_eq.setScalar(0)
    a_output_eq.dump()
    network.addEquation(a_output_eq)

    # B = zkf <--> B - z_k_f = 0
    b_output_eq = MarabouCore.Equation()
    b_output_eq.addAddend(1, b_idx)
    b_output_eq.addAddend(-b_w, z_i_f_idx)
    b_output_eq.setScalar(0)
    b_output_eq.dump()
    network.addEquation(b_output_eq)

    min_b = relu(relu(xlim[0][0] * x0_z_w) + relu(xlim[1][0] * x1_z_w) * b_w)
    max_b = relu(relu(xlim[0][1] * x0_z_w) + relu(xlim[1][1] * x1_z_w) * b_w)

    min_a = relu(relu(xlim[0][0] * x0_s_w) + relu(xlim[1][0] * x1_s_w) * a_w)
    max_a = relu(relu(xlim[0][1] * x0_s_w) + relu(xlim[1][1] * x1_s_w) * a_w)

    initial_diff = min_a - max_b
    # assert initial_diff >= 0
    alpha = initial_diff / (2 * n_iterations) + max_b
    print('min_a', min_a)
    print('max_b', max_b)
    print('initial_diff', initial_diff)
    print('alpha', alpha)

    a_invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    a_invariant_equation.addAddend(1, s_i_f_idx)  # a_i
    a_invariant_equation.addAddend(alpha, s_cell_iterator)  # i
    a_invariant_equation.setScalar(min_a)

    b_invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    b_invariant_equation.addAddend(1, z_i_f_idx)  # a_i
    b_invariant_equation.addAddend(-alpha, z_cell_iterator)  # i
    b_invariant_equation.setScalar(max_b)

    return network, [s_cell_iterator, z_cell_iterator], [a_invariant_equation, b_invariant_equation], \
           (min_a, max_b), (-alpha, alpha)


def define_adversarial_robustness_two_input_nodes(xlim, n_iterations):
    '''
    Define an adversarial roustness examples
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
    :return: network, [s_cell_iterator, z_cell_iterator], [a_invariant_equation, b_invariant_equation],\
           (min_a, max_b), [a_base_equation, b_base_equation], (-alpha, alpha)
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
    x1_s_w = 5
    x0_z_w = 2
    x1_z_w = 1

    # s_i_f = relu(2 * x1 + 1 * x2 + 1.5*s_i-1_f)
    s_cell_iterator = network.getNumberOfVariables()  # i
    s_i_f_idx = add_rnn_cell(network, [(0, x0_s_w), (1, x1_s_w)], s_hidden_w, n_iterations, print_debug=True)
    s_i_1_f_idx = s_i_f_idx - 2
    # z_i_f = relu(1 * x1 + 10 * x2 + z_i-1_f)
    z_cell_iterator = network.getNumberOfVariables()
    z_i_f_idx = add_rnn_cell(network, [(0, x0_z_w), (1, x1_z_w)], z_hidden_w, n_iterations, print_debug=True)
    z_i_1_f_idx = z_i_f_idx - 2

    a_idx = z_i_f_idx + 1
    b_idx = a_idx + 1

    a_w = 1
    b_w = 1

    network.setNumberOfVariables(b_idx + 1)

    # A
    network.setLowerBound(a_idx, -large)  # A
    network.setUpperBound(a_idx, large)

    # B
    network.setLowerBound(b_idx, -large)  # B
    network.setUpperBound(b_idx, large)

    # # A = skf <--> A - skf = 0
    a_output_eq = MarabouCore.Equation()
    a_output_eq.addAddend(1, a_idx)
    a_output_eq.addAddend(-a_w, s_i_f_idx)
    a_output_eq.setScalar(0)
    a_output_eq.dump()
    network.addEquation(a_output_eq)

    # B = zkf <--> B - z_k_f = 0
    b_output_eq = MarabouCore.Equation()
    b_output_eq.addAddend(1, b_idx)
    b_output_eq.addAddend(-b_w, z_i_f_idx)
    b_output_eq.setScalar(0)
    b_output_eq.dump()
    network.addEquation(b_output_eq)

    min_b = relu(relu(xlim[0][0] * x0_z_w) + relu(xlim[1][0] * x1_z_w) * b_w)
    max_b = relu(relu(xlim[0][1] * x0_z_w) + relu(xlim[1][1] * x1_z_w) * b_w)

    min_a = relu(relu(xlim[0][0] * x0_s_w) + relu(xlim[1][0] * x1_s_w) * a_w)
    max_a = relu(relu(xlim[0][1] * x0_s_w) + relu(xlim[1][1] * x1_s_w) * a_w)

    initial_diff = min_a - max_b
    # assert initial_diff >= 0
    alpha = initial_diff / (2 * n_iterations)
    print('min_a:', min_a)
    print('max_b:', max_b)
    print('initial_diff:', initial_diff)
    print('alpha:', alpha)

    # a_i >= a_0 - (alpha + max_b) * i
    a_invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    a_invariant_equation.addAddend(1, s_i_f_idx)  # a_i
    a_invariant_equation.addAddend(alpha, s_cell_iterator)  # i
    a_invariant_equation.setScalar(min_a)
    # invariant_equation.dump()

    # b_i <= b_0 + (alpha + max_b) * i
    b_invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    b_invariant_equation.addAddend(1, z_i_f_idx)  # a_i
    b_invariant_equation.addAddend(-alpha, z_cell_iterator)  # i
    b_invariant_equation.setScalar(max_b)

    return network, [s_cell_iterator, z_cell_iterator], [a_invariant_equation, b_invariant_equation], \
           (min_a, max_b), (-alpha, alpha)


def define_weak_adversarial_robustness(xlim, n_iterations):
    '''
    Defines adversarial robustness where the property cannot be derived from the invariants.
    We expect that the invariant will hold but the property will not
    :param xlim:
    :param n_iterations:
    :return:
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
    x1_s_w = 5
    x0_z_w = 2
    x1_z_w = 1

    # s_i_f = relu(2 * x1 + 1 * x2 + 1.5*s_i-1_f)
    s_cell_iterator = network.getNumberOfVariables()  # i
    s_i_f_idx = add_rnn_cell(network, [(0, x0_s_w), (1, x1_s_w)], s_hidden_w, n_iterations, print_debug=True)
    s_i_1_f_idx = s_i_f_idx - 2
    # z_i_f = relu(1 * x1 + 10 * x2 + z_i-1_f)
    z_cell_iterator = network.getNumberOfVariables()
    z_i_f_idx = add_rnn_cell(network, [(0, x0_z_w), (1, x1_z_w)], z_hidden_w, n_iterations, print_debug=True)
    z_i_1_f_idx = z_i_f_idx - 2

    a_idx = z_i_f_idx + 1
    b_idx = a_idx + 1

    a_w = 1
    b_w = 1

    network.setNumberOfVariables(b_idx + 1)

    # A
    network.setLowerBound(a_idx, -large)  # A
    network.setUpperBound(a_idx, large)

    # B
    network.setLowerBound(b_idx, -large)  # B
    network.setUpperBound(b_idx, large)

    # # A = skf <--> A - skf = 0
    a_output_eq = MarabouCore.Equation()
    a_output_eq.addAddend(1, a_idx)
    a_output_eq.addAddend(-a_w, s_i_f_idx)
    a_output_eq.setScalar(0)
    a_output_eq.dump()
    network.addEquation(a_output_eq)

    # B = zkf <--> B - z_k_f = 0
    b_output_eq = MarabouCore.Equation()
    b_output_eq.addAddend(1, b_idx)
    b_output_eq.addAddend(-b_w, z_i_f_idx)
    b_output_eq.setScalar(0)
    b_output_eq.dump()
    network.addEquation(b_output_eq)

    min_b = relu(relu(xlim[0][0] * x0_z_w) + relu(xlim[1][0] * x1_z_w) * b_w)
    max_b = relu(relu(xlim[0][1] * x0_z_w) + relu(xlim[1][1] * x1_z_w) * b_w)

    min_a = relu(relu(xlim[0][0] * x0_s_w) + relu(xlim[1][0] * x1_s_w) * a_w)
    max_a = relu(relu(xlim[0][1] * x0_s_w) + relu(xlim[1][1] * x1_s_w) * a_w)

    initial_diff = min_a - max_b
    # assert initial_diff >= 0
    alpha = initial_diff / (n_iterations * 2) + max_b
    print('min_a', min_a)
    print('max_b', max_b)
    print('initial_diff', initial_diff)
    print('alpha', alpha)

    # a_i - b_i >= a_i-1 - b_i-1 + alpha <--> a_i - a_i-1 - b_i + b_i-1 >= alpha
    a_invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
    a_invariant_equation.addAddend(1, a_idx)  # a_i
    a_invariant_equation.addAddend(alpha, s_cell_iterator)  # i
    a_invariant_equation.setScalar(min_a)
    # invariant_equation.dump()

    b_invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    b_invariant_equation.addAddend(1, b_idx)  # a_i
    b_invariant_equation.addAddend(-alpha, z_cell_iterator)  # i
    b_invariant_equation.setScalar(max_b)

    return network, [s_cell_iterator, z_cell_iterator], [a_invariant_equation, b_invariant_equation], \
           (min_a, max_b), (alpha, alpha + 100)


def define_adversarial_robustness_one_input(xlim, n_iterations):
    '''
    Define an adversarial roustness examples
    0 <= x_0 <= 1 # actually it's acording to xlim
    s_i = 6 * x_0 + 0.5 * s_(i-1)
    z_i = 1 * x_0 + 1 * z_(i-1)
    A = s_i
    B = z_i

    prove that after n_iterations A >= B
    :param xlim: array of tuples, each array cell as an input (x_0, x_1 etc..) tuple is (min_value, max_value)
    :param n_iterations: number of iterations
    :return: network, [s_cell_iterator, z_cell_iterator], [a_invariant_equation, b_invariant_equation],\
           (min_a, max_b), [a_base_equation, b_base_equation], (-alpha, alpha)
    '''
    network = MarabouCore.InputQuery()
    network.setNumberOfVariables(len(xlim))  # x1

    # x1
    network.setLowerBound(0, xlim[0][0])
    network.setUpperBound(0, xlim[0][1])

    s_hidden_w = 0.5
    z_hidden_w = 1
    x0_s_w = 6
    x0_z_w = 1

    s_cell_iterator = network.getNumberOfVariables()  # i
    s_i_f_idx = add_rnn_cell(network, [(0, x0_s_w)], s_hidden_w, n_iterations, print_debug=True)
    s_i_1_f_idx = s_i_f_idx - 2

    z_cell_iterator = network.getNumberOfVariables()
    z_i_f_idx = add_rnn_cell(network, [(0, x0_z_w)], z_hidden_w, n_iterations, print_debug=True)
    z_i_1_f_idx = z_i_f_idx - 2

    a_idx = z_i_f_idx + 1
    b_idx = a_idx + 1

    a_w = 1
    b_w = 1

    network.setNumberOfVariables(b_idx + 1)

    # A
    network.setLowerBound(a_idx, -large)  # A
    network.setUpperBound(a_idx, large)

    # B
    network.setLowerBound(b_idx, -large)  # B
    network.setUpperBound(b_idx, large)

    # # A = skf <--> A - skf = 0
    a_output_eq = MarabouCore.Equation()
    a_output_eq.addAddend(1, a_idx)
    a_output_eq.addAddend(-a_w, s_i_f_idx)
    a_output_eq.setScalar(0)
    a_output_eq.dump()
    network.addEquation(a_output_eq)

    # B = zkf <--> B - z_k_f = 0
    b_output_eq = MarabouCore.Equation()
    b_output_eq.addAddend(1, b_idx)
    b_output_eq.addAddend(-b_w, z_i_f_idx)
    b_output_eq.setScalar(0)
    b_output_eq.dump()
    network.addEquation(b_output_eq)

    min_b = relu(relu(xlim[0][0] * x0_z_w) * b_w)
    max_b = relu(relu(xlim[0][1] * x0_z_w) * b_w)

    min_a = relu(relu(xlim[0][0] * x0_s_w) * a_w)
    max_a = relu(relu(xlim[0][1] * x0_s_w) * a_w)

    return network, [s_cell_iterator, z_cell_iterator], None, (min_a, max_b)


def define_adversarial_robustness_concatenate_rnn_fully_connect(xlim, n_iterations):
    '''
    xlim[0] <= x_0 <= xlim[1]
    s1_i = 10 * x_0 + 0.5 * s1_i-1
    s2_i = 1 * s1_i + 1 * z1_i + 1 * s2_i-1
    z1_i = 1 * x_0 + 1 * z1_i-1 + 1 * z1_i-1
    z2_i = 1 * z1_i + 0.9 * s1_i + 1 * z2_i-1
    A = s2
    B = z2
    '''
    query = MarabouCore.InputQuery()
    query.setNumberOfVariables(len(xlim))  # x

    # x
    query.setLowerBound(0, xlim[0][0])
    query.setUpperBound(0, xlim[0][1])

    # s1
    w_x_s1 = 1
    w_s1_s1 = 1

    # z1
    w_x_z1 = 1
    w_z1_z1 = 1

    # s2
    w_s1_s2 = 10
    w_s2_s2 = 0.5
    w_z1_s2 = 1

    # z2
    w_s1_z2 = 1
    w_z1_z2 = 1
    w_z2_z2 = 1

    w_z2_b = 1
    w_s2_a = 1
    s1_out_idx = add_rnn_cell(query, [(0, w_x_s1)], w_s1_s1, n_iterations)
    z1_out_idx = add_rnn_cell(query, [(0, w_x_z1)], w_z1_z1, n_iterations)

    s2_out_idx = add_rnn_cell(query, [(s1_out_idx, w_s1_s2), (z1_out_idx, w_z1_s2)], w_s2_s2, n_iterations)
    z2_out_idx = add_rnn_cell(query, [(z1_out_idx, w_z1_z2), (s1_out_idx, w_s1_z2)], w_z2_z2, n_iterations)

    a_idx = z2_out_idx + 1
    b_idx = a_idx + 1
    print("s1_out_idx:", s1_out_idx)
    print("s2_out_idx:", s2_out_idx)
    print("z1_out_idx:", z1_out_idx)
    print("z2_out_idx:", z2_out_idx)
    print("a_idx:", a_idx)
    print("b_idx:", b_idx)

    query.setNumberOfVariables(b_idx + 1)

    query.setLowerBound(a_idx, -large)
    query.setUpperBound(a_idx, large)
    query.setLowerBound(b_idx, -large)
    query.setUpperBound(b_idx, large)

    a_output_equation = MarabouCore.Equation()
    a_output_equation.addAddend(1, a_idx)
    a_output_equation.addAddend(-w_s2_a, s2_out_idx)
    a_output_equation.setScalar(0)
    query.addEquation(a_output_equation)

    b_output_equation = MarabouCore.Equation()
    b_output_equation.addAddend(1, b_idx)
    b_output_equation.addAddend(-w_z2_b, z2_out_idx)
    b_output_equation.setScalar(0)
    query.addEquation(b_output_equation)

    min_s1 = relu(xlim[0][0] * w_x_s1)
    max_s1 = relu(xlim[0][1] * w_x_s1)
    min_s2 = relu(relu(xlim[0][0] * w_x_s1) * w_s1_s2 + relu(xlim[0][0] * w_x_z1) * w_z1_s2)
    min_z1 = relu(xlim[0][0] * w_x_z1)
    max_z1 = relu(xlim[0][1] * w_x_z1)
    max_z2 = relu(relu(xlim[0][1] * w_x_z1) * w_z1_z2 + relu(xlim[0][1] * w_x_s1) * w_s1_z2)

    # min_a = relu(relu(relu(xlim[0][0] * w_x_s1) * w_s1_s2) * w_s2_a)
    # max_a = relu(relu(relu(xlim[0][1] * w_x_s1) * w_s1_s2) * w_s2_a)
    # min_b = relu(relu(relu(xlim[0][0] * w_x_z1) * w_z1_z2) * w_z2_b)
    # max_b = relu(relu(relu(xlim[0][1] * w_x_z1) * w_z1_z2) * w_z2_b)
    # print('min_a', min_a)
    # print('max_a', max_a)
    # print('min_b', min_b)
    # print('max_b', max_b)

    # This means that the only cell that is dependent on s1 is s2, and same for z1 and z2
    rnn_dependent = [[4, 5], [4, 5], [4, 5], [4, 5], None, None]
    return query, [i - 3 for i in [s1_out_idx, s1_out_idx, z1_out_idx, z1_out_idx, s2_out_idx, z2_out_idx]], None, (
        min_s1, max_s1, min_z1, max_z1, min_s2, max_z2), rnn_dependent


def define_adversarial_robustness_concatenate_rnn(xlim, n_iterations):
    '''
    xlim[0] <= x_0 <= xlim[1]
    s1_i = 10 * x_0 + 0.5 * s1_i-1
    s2_i = 1 * s1_i + 1 * s2_i-1
    z1_i = 1 * x_0 + 1 * z1_i-1
    z2_i = 1 * z1_i + 1 * z2_i-1
    A = s2
    B = z2
    indcies:
        0: x_0
        1: i
        2: s1_i-1
        3: s1_i_b
        4: s1_i_f
        5: i
        6: z1_i-1
        7: z1_i_b
        8: z1_i_f
        9: i
        10: s2_i-1
        11: s2_i_b
        12: s2_i_f
        13: i
        14: z2_i-1
        15: z2_i_b
        16: z2_i_f
        17: A
        18: B
    '''
    query = MarabouCore.InputQuery()
    query.setNumberOfVariables(len(xlim))  # x

    # x
    query.setLowerBound(0, xlim[0][0])
    query.setUpperBound(0, xlim[0][1])

    w_x_s1 = 10
    w_s1_s2 = 1
    w_s1_s1 = 0.5
    w_s2_s2 = 1
    w_s2_a = 1

    w_x_z1 = 1
    w_z1_z2 = 1
    w_z1_z1 = 1
    w_z2_z2 = 1
    w_z2_b = 1

    s1_out_idx = add_rnn_cell(query, [(0, w_x_s1)], w_s1_s1, n_iterations)
    z1_out_idx = add_rnn_cell(query, [(0, w_x_z1)], w_z1_z1, n_iterations)

    s2_out_idx = add_rnn_cell(query, [(s1_out_idx, w_s1_s2)], w_s2_s2, n_iterations)
    z2_out_idx = add_rnn_cell(query, [(z1_out_idx, w_z1_z2)], w_z2_z2, n_iterations)

    a_idx = z2_out_idx + 1
    b_idx = a_idx + 1
    print("s1_out_idx:", s1_out_idx)
    print("s2_out_idx:", s2_out_idx)
    print("z1_out_idx:", z1_out_idx)
    print("z2_out_idx:", z2_out_idx)
    print("a_idx:", a_idx)
    print("b_idx:", b_idx)

    query.setNumberOfVariables(b_idx + 1)

    query.setLowerBound(a_idx, -large)
    query.setUpperBound(a_idx, large)
    query.setLowerBound(b_idx, -large)
    query.setUpperBound(b_idx, large)

    a_output_equation = MarabouCore.Equation()
    a_output_equation.addAddend(1, a_idx)
    a_output_equation.addAddend(-w_s2_a, s2_out_idx)
    a_output_equation.setScalar(0)
    query.addEquation(a_output_equation)

    b_output_equation = MarabouCore.Equation()
    b_output_equation.addAddend(1, b_idx)
    b_output_equation.addAddend(-w_z2_b, z2_out_idx)
    b_output_equation.setScalar(0)
    query.addEquation(b_output_equation)

    min_s1 = relu(xlim[0][0] * w_x_s1)
    min_s2 = relu(relu(xlim[0][0] * w_x_s1) * w_s1_s2)
    max_z1 = relu(xlim[0][1] * w_x_z1)
    max_z2 = relu(relu(xlim[0][1] * w_x_z1) * w_z1_z2)

    # min_a = relu(relu(relu(xlim[0][0] * w_x_s1) * w_s1_s2) * w_s2_a)
    # max_a = relu(relu(relu(xlim[0][1] * w_x_s1) * w_s1_s2) * w_s2_a)
    # min_b = relu(relu(relu(xlim[0][0] * w_x_z1) * w_z1_z2) * w_z2_b)
    # max_b = relu(relu(relu(xlim[0][1] * w_x_z1) * w_z1_z2) * w_z2_b)
    # print('min_a', min_a)
    # print('max_a', max_a)
    # print('min_b', min_b)
    # print('max_b', max_b)

    # This means that the only cell that is dependent on s1 is s2, and same for z1 and z2
    rnn_dependent = [[2], [3], None, None]
    return query, [i - 3 for i in [s1_out_idx, z1_out_idx, s2_out_idx, z2_out_idx]], None, (
        min_s1, max_z1, min_s2, max_z2), rnn_dependent


def test_auto_adversarial_robustness_one_input():
    '''
    This exmple has only one input node and two RNN cells
    '''
    n_iterations = 4
    xlim = [(1, 2)]

    network, rnn_start_idxs, _, initial_values, *_ = define_adversarial_robustness_one_input(xlim,
                                                                                             n_iterations)
    rnn_invariant_type = [MarabouCore.Equation.GE, MarabouCore.Equation.LE]
    inv_res = find_invariant_adversarial(network, rnn_start_idxs, rnn_invariant_type, initial_values, n_iterations)
    print(inv_res)
    assert inv_res


def test_auto_adversarial_robustness_one_input_fail():
    num_iterations = 5  # This is suppose to be true only for n >= 12
    xlim = [(1, 2)]

    network, rnn_start_idxs, _, initial_values, *_ = define_adversarial_robustness_one_input(xlim,
                                                                                             num_iterations)
    rnn_invariant_type = [MarabouCore.Equation.GE, MarabouCore.Equation.LE]
    inv_res = find_invariant_adversarial(network, rnn_start_idxs, rnn_invariant_type, initial_values, num_iterations)
    print(inv_res)
    assert not inv_res


def test_auto_adversarial_robustness_two_inputs():
    '''
    This example has 2 input nodes and two RNN cells
    '''
    num_iterations = 10
    xlim = [(0, 1), (1, 2)]

    network, rnn_start_idxs, _, initial_values, *_ = define_adversarial_robustness_two_input_nodes(xlim, num_iterations)
    rnn_invariant_type = [MarabouCore.Equation.GE, MarabouCore.Equation.LE]
    inv_res = find_invariant_adversarial(network, rnn_start_idxs, rnn_invariant_type, initial_values, num_iterations)
    assert inv_res


def test_auto_adversarial_robustness_two_inputs_fail():
    num_iterations = 10
    xlim = [(0, 1), (1, 2)]
    # y_lim = 10 ** -2

    network, rnn_start_idxs, _, initial_values, *_ = define_adversarial_robustness_two_input_nodes_step_fail(xlim,
                                                                                                             num_iterations)
    rnn_invariant_type = [MarabouCore.Equation.GE, MarabouCore.Equation.LE]
    inv_res = find_invariant_adversarial(network, rnn_start_idxs, rnn_invariant_type, initial_values, num_iterations)
    assert not inv_res

    # previously didn't test this with auto, just used the fixed invariant, here is how to run the previous experiment
    # assert not prove_adversarial_using_invariant(xlim, num_iterations,
    #                                              define_adversarial_robustness_two_input_nodes_step_fail)


def test_adversarial_robustness_two_inputs_base_fail():
    num_iterations = 10
    xlim = [(0, 10), (1, 2)]
    # y_lim = 10 ** -2

    assert not prove_adversarial_using_invariant(xlim, num_iterations,
                                                 define_adversarial_robustness_two_input_nodes)


def test_adversarial_robustness_conclusion_fail():
    num_iterations = 100
    xlim = [(0, 1), (1, 2)]
    # y_lim = 10 ** -2

    assert not prove_adversarial_using_invariant(xlim, num_iterations, define_weak_adversarial_robustness)


def test_z3_adversarial_robustness():
    a_pace = -1
    b_pace = 1
    min_a = 5
    max_b = 0
    n_iterations = 2
    assert prove_adversarial_property_z3(a_pace, b_pace, min_a, max_b, n_iterations)


def test_z3_adversarial_robustness_fail():
    a_pace = -1
    b_pace = 1
    min_a = 5
    max_b = 0
    n_iterations = 100
    assert not prove_adversarial_property_z3(a_pace, b_pace, min_a, max_b, n_iterations)


def test_invariant_bounds_ge():
    num_iterations = 100

    def define_A_network(xlim, n_iterations, hidden_weight):
        '''
        Define an adversarial robustness examples
        1 <= x_0 <= 2
        s_i = 10 * x_0 +  - hidden_weight * s_(i-1)
        A = s_i
        B = 10
        prove that after n_iterations A >= B
        :param xlim: array of tuples, each array cell as an input (x_0, x_1 etc..) tuple is (min_value, max_value)
        :param n_iterations: number of iterations
        :return: network, [s_cell_iterator, z_cell_iterator], [a_invariant_equation, b_invariant_equation],\
               (min_a, max_b), [a_base_equation, b_base_equation], (-alpha, alpha)
        '''
        network = MarabouCore.InputQuery()
        network.setNumberOfVariables(len(xlim))  # x1, x2

        # x1
        network.setLowerBound(0, xlim[0][0])
        network.setUpperBound(0, xlim[0][1])

        s_hidden_w = hidden_weight
        x0_s_w = 10
        x1_s_w = 1

        s_cell_iterator = network.getNumberOfVariables()  # i
        s_i_f_idx = add_rnn_cell(network, [(0, x0_s_w)], s_hidden_w, n_iterations, print_debug=True)

        a_idx = s_i_f_idx + 1
        b_idx = a_idx + 1

        b_value = 5
        a_w = 1
        b_w = 1

        network.setNumberOfVariables(b_idx + 1)

        # A
        network.setLowerBound(a_idx, -large)  # A
        network.setUpperBound(a_idx, large)

        # B
        network.setLowerBound(b_idx, -large)  # B
        network.setUpperBound(b_idx, large)

        b_fix_val = MarabouCore.Equation()
        b_fix_val.addAddend(1, b_idx)
        b_fix_val.setScalar(b_value)
        b_fix_val.dump()
        network.addEquation(b_fix_val)

        # B = zkf <--> B - z_k_f = 0
        a_output_eq = MarabouCore.Equation()
        a_output_eq.addAddend(1, a_idx)
        a_output_eq.addAddend(-a_w, s_i_f_idx)
        a_output_eq.setScalar(0)
        a_output_eq.dump()
        network.addEquation(a_output_eq)

        min_b = b_value  # relu(relu(xlim[0][0] * x0_s_w) + relu(xlim[1][0] * x1_s_w) * b_w)
        max_b = b_value  # relu(relu(xlim[0][1] * x0_s_w) + relu(xlim[1][1] * x1_s_w) * b_w)

        min_a = relu(relu(xlim[0][0] * x0_s_w) * a_w)
        max_a = relu(relu(xlim[0][1] * x0_s_w) * a_w)

        return network, [s_cell_iterator], None, (min_a, max_b), None

    xlim = [(1, 2)]
    alpha = 2  # 20 # Change to 20 and then we will prove all weights which is wrong
    good_weight = []
    fail_weight = []
    for weight in [-2, -1, -0.5, 0, 1, 2]:
        partial_define = lambda xlim, ylim, n_iterations: define_A_network(xlim, n_iterations, weight)
        network, rnn_start_idxs, invariant_equation, initial_values, _ = partial_define(xlim, None, num_iterations)
        invariant_equation = MarabouCore.Equation(MarabouCore.Equation.GE)
        invariant_equation.addAddend(1, rnn_start_idxs[0] + 3)  # a_i
        invariant_equation.addAddend(alpha, rnn_start_idxs[0])  # i
        invariant_equation.setScalar(initial_values[0])
        if prove_invariant2(network, rnn_start_idxs, [invariant_equation]):
            good_weight.append(weight)
        else:
            fail_weight.append(weight)
    print('successfully proved on weights:', good_weight)
    print("couldn't proved on weights:", fail_weight)
    assert len([w for w in fail_weight if w >= 0]) == 0

    # network, rnn_start_idxs, invariant_equation, initial_values, _ = define_B_gets_smaller(
    #     invariant_xlim, num_iterations)
    # fail_alpha = []
    # for alpha in range(bounds[0] * 10, bounds[1] * 10, 1):
    #     alpha = alpha / 10
    #     invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    #     invariant_equation.addAddend(1, rnn_start_idxs[0] + 3)  # b_i
    #     invariant_equation.addAddend(-alpha, rnn_start_idxs[0])  # i
    #     invariant_equation.setScalar(initial_values[1])
    #
    #     partial_define = lambda xlim, ylim, n_iterations: define_B_gets_smaller(xlim, n_iterations)
    #     if not prove_invariant2(partial_define, [invariant_equation], invariant_xlim, num_iterations):
    #         fail_alpha.append(alpha)

    # print("Fail alpha:\n", fail_alpha)/


def test_invariant_bounds_le():
    num_iterations = 6

    def define_B_network(xlim, n_iterations, hidden_weight):
        '''
        Define an adversarial robustness examples
        1 <= x_0 <= 2
        s_i = 10 * x_0 +  - hidden_weight * s_(i-1)
        A = s_i
        B = 10
        prove that after n_iterations A >= B
        :param xlim: array of tuples, each array cell as an input (x_0, x_1 etc..) tuple is (min_value, max_value)
        :param n_iterations: number of iterations
        :return: network, [s_cell_iterator, z_cell_iterator], [a_invariant_equation, b_invariant_equation],\
               (min_a, max_b), [a_base_equation, b_base_equation], (-alpha, alpha)
        '''
        network = MarabouCore.InputQuery()
        network.setNumberOfVariables(len(xlim))  # x1, x2

        # x1
        network.setLowerBound(0, xlim[0][0])
        network.setUpperBound(0, xlim[0][1])

        s_hidden_w = hidden_weight
        x0_s_w = 10

        s_cell_iterator = network.getNumberOfVariables()  # i
        s_i_f_idx = add_rnn_cell(network, [(0, x0_s_w)], s_hidden_w, n_iterations, print_debug=True)

        a_idx = s_i_f_idx + 1
        b_idx = a_idx + 1

        a_value = 5
        a_w = 1
        b_w = 1

        network.setNumberOfVariables(b_idx + 1)

        # A
        network.setLowerBound(a_idx, -large)  # A
        network.setUpperBound(a_idx, large)

        # B
        network.setLowerBound(b_idx, -large)  # B
        network.setUpperBound(b_idx, large)

        a_fix_val = MarabouCore.Equation()
        a_fix_val.addAddend(1, a_idx)
        a_fix_val.setScalar(a_value)
        a_fix_val.dump()
        network.addEquation(a_fix_val)

        # B = zkf <--> B - z_k_f = 0
        b_output_eq = MarabouCore.Equation()
        b_output_eq.addAddend(1, b_idx)
        b_output_eq.addAddend(-b_w, s_i_f_idx)
        b_output_eq.setScalar(0)
        b_output_eq.dump()
        network.addEquation(b_output_eq)

        min_b = relu(relu(xlim[0][0] * x0_s_w) * b_w)
        max_b = relu(relu(xlim[0][1] * x0_s_w) * b_w)

        min_a = a_value
        max_a = a_value

        return network, [s_cell_iterator], None, (min_a, max_b), None

    xlim = [(1, 2)]
    alpha = 100
    good_weight = []
    fail_weight = []
    for weight in [1.1]:  # , -1, -0.5, 0, 0.5, 1, 1.1, 1.5, 2]:
        partial_define = lambda xlim, ylim, n_iterations: define_B_network(xlim, n_iterations, weight)
        network, rnn_start_idxs, invariant_equation, initial_values, _ = partial_define(xlim, None, num_iterations)
        invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
        invariant_equation.addAddend(1, rnn_start_idxs[0] + 3)  # a_i
        invariant_equation.addAddend(-alpha, rnn_start_idxs[0])  # i
        invariant_equation.setScalar(initial_values[1])
        if prove_invariant2(network, rnn_start_idxs, [invariant_equation]):
            good_weight.append(weight)
        else:
            fail_weight.append(weight)
    print('successfully proved on weights:', good_weight)
    print("couldn't proved on weights:", fail_weight)
    assert len([w for w in fail_weight if w < 1 and w > -1]) == 0

    # network, rnn_start_idxs, invariant_equation, initial_values, _ = define_B_gets_smaller(
    #     invariant_xlim, num_iterations)
    # fail_alpha = []
    # for alpha in range(bounds[0] * 10, bounds[1] * 10, 1):
    #     alpha = alpha / 10
    #     invariant_equation = MarabouCore.Equation(MarabouCore.Equation.LE)
    #     invariant_equation.addAddend(1, rnn_start_idxs[0] + 3)  # b_i
    #     invariant_equation.addAddend(-alpha, rnn_start_idxs[0])  # i
    #     invariant_equation.setScalar(initial_values[1])
    #
    #     partial_define = lambda xlim, ylim, n_iterations: define_B_gets_smaller(xlim, n_iterations)
    #     if not prove_invariant2(partial_define, [invariant_equation], invariant_xlim, num_iterations):
    #         fail_alpha.append(alpha)

    # print("Fail alpha:\n", fail_alpha)


def test_auto_adversarial_robustness_one_input_concatenate_rnns():
    '''
    This exmple has only one input node and two RNN cells
    '''
    # return
    num_iterations = 5
    xlim = [(1, 2)]

    network, rnn_start_idxs, _, initial_values, rnn_dependent = define_adversarial_robustness_concatenate_rnn(xlim,
                                                                                                              num_iterations)
    rnn_invariant_type = [MarabouCore.Equation.GE, MarabouCore.Equation.LE, MarabouCore.Equation.GE,
                          MarabouCore.Equation.LE]
    inv_res = find_invariant_adversarial(network, rnn_start_idxs, rnn_invariant_type, initial_values, num_iterations,
                                         rnn_dependent=rnn_dependent)

    print(inv_res)
    assert inv_res


def test_auto_adversarial_robustness_one_input_concatenate_rnns_fc():
    '''
    This exmple has only one input node and two RNN cells
    '''
    # return
    num_iterations = 3
    xlim = [(1, 2)]

    network, rnn_start_idxs, _, initial_values, rnn_dependent = define_adversarial_robustness_concatenate_rnn_fully_connect(
        xlim, num_iterations)

    rnn_invariant_type = [MarabouCore.Equation.GE, MarabouCore.Equation.LE, MarabouCore.Equation.GE,
                          MarabouCore.Equation.LE, MarabouCore.Equation.GE, MarabouCore.Equation.LE]
    inv_res = find_invariant_adversarial(network, rnn_start_idxs, rnn_invariant_type, initial_values, num_iterations,
                                         rnn_dependent=rnn_dependent)

    print(inv_res)
    assert inv_res


def test_auto_adversarial_robustness_one_input_concatenate_rnns_fail():
    '''
    This exmple has only one input node and two RNN cells
    '''
    # return
    num_iterations = 6
    xlim = [(1, 2)]

    network, rnn_start_idxs, _, initial_values, rnn_dependent = define_adversarial_robustness_concatenate_rnn(xlim,
                                                                                                              num_iterations)
    rnn_invariant_type = [MarabouCore.Equation.GE, MarabouCore.Equation.LE, MarabouCore.Equation.GE,
                          MarabouCore.Equation.LE]
    inv_res = find_invariant_adversarial(network, rnn_start_idxs, rnn_invariant_type, initial_values, num_iterations,
                                         rnn_dependent=rnn_dependent)

    print(inv_res)
    assert not inv_res


def test_auto_adversarial_robustness_two_inputs_SGDAlgorithm():
    '''
    This example has 2 input nodes and two RNN cells
    '''
    from RNN.MarabouRNNMultiDim import prove_multidim_property
    from rnn_algorithms.IterateAlphasSGD import IterateAlphasSGD

    num_iterations = 10
    xlim = [(0, 1), (1, 2)]

    network, rnn_start_idxs, property_eq, initial_values, *_ = define_adversarial_robustness_two_input_nodes(xlim, num_iterations)
    rnn_invariant_type = [MarabouCore.Equation.GE, MarabouCore.Equation.LE]


    # network.dump()
    rnn_output_idxs = [i + 3 for i in rnn_start_idxs]
    algorithm = IterateAlphasSGD(initial_values, rnn_start_idxs, rnn_output_idxs)
    assert prove_multidim_property(network, rnn_start_idxs, rnn_output_idxs, property_eq, algorithm)
