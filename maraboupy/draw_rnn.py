import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from tqdm import tqdm

MIN_GRID = -2
MAX_GRID = 17
NUM_POINTS_TO_SAMPLE = 500
EPS = 10 ** -4


def ReLU(x):
    '''
    return element wise ReLU (max between 0,x)
    :param x: int or np.ndarray
    :return: same type, only positive numbers
    '''
    if isinstance(x, int) or isinstance(x, np.float64):
        return max(0, x)
    elif isinstance(x, np.ndarray):
        # if len(x.shape) == 1:
        #     x = x[:, None]
        return np.vectorize(lambda v: max(0.0, v))(x)
    else:
        return None


def calc_rnn_values(x: int, w_in: np.ndarray, w_h: np.ndarray, num_steps: int):
    '''
    Calc the linear value of the rnn
    :param x: input variable
    :param w_in: 1d of weights, shape = d
    :param w_h: matrix of hidden weight, shape d*d
    :param num_steps: number of steps to run
    :return: rnn values, shape is num_steps*d, i.e. every column is the output of one dimension
    '''
    d = w_h.shape[0]
    r = np.zeros((num_steps + 1, d))

    for i in range(1, num_steps + 1):
        r[i, ...] = ReLU(np.matmul(x, w_in) + np.matmul(w_h, r[i - 1, ...]))
        # r[i, ...] = x * w_in + np.matmul(w _h, r[i - 1, ...])

    return r  # [1:, :]


def draw_r_values(r0_values, r1_values, eps=None):
    if eps:
        def get_alpha(r_values, eps):
            # i = 0
            return [(r_values[i] + eps) / (1) for i in range(len(r_values))]

        alpha_0_min = get_alpha(r0_values, -eps)
        alpha_0_max = get_alpha(r0_values, eps)
        alpha_1_min = get_alpha(r1_values, -eps)
        alpha_1_max = get_alpha(r1_values, eps)

        top_left_points = list(zip(alpha_0_min, alpha_1_max))
        top_right_points = list(zip(alpha_0_max, alpha_1_max))
        bottom_left_points = list(zip(alpha_0_min, alpha_1_min))
        bottom_right_points = list(zip(alpha_0_max, alpha_1_min))
        box_params = {'c': 'red', 'alpha': 0.8}
        # plt.scatter(*zip(*top_left_points),     **box_params)
        # plt.scatter(*zip(*top_right_points),    **box_params)
        # plt.scatter(*zip(*bottom_left_points),  **box_params)
        # plt.scatter(*zip(*bottom_right_points), **box_params)

    plt.scatter(r0_values, r1_values)
    plt.xlabel('r0')
    plt.ylabel('r1')
    plt.savefig("figures/rnn_values_wh:{}.png".format(w_h))
    print(list(zip(r0_values, r1_values)))


def draw_min_alphas(x: int, w_in: np.ndarray, w_h: np.ndarray, num_steps: int):
    r = calc_rnn_values(x, w_in, w_h, num_steps)
    r0 = r[:, 0]
    r1 = r[:, 1]
    d = np.linspace(MIN_GRID, MAX_GRID, 5000)
    a0, a1 = np.meshgrid(d, d)

    a0_constraints = []
    for i, v in enumerate(r0):
        a0_constraints.append(a0 * i + x * w_in[0] <= v)
        a0_constraints.append(a0 * i <= a0 * (i - 1) * w_h[0, 0] + a1 * (i - 1) * w_h[0, 1])

    a1_constraints = []
    for i, v in enumerate(r1):
        a1_constraints.append(a1 * i + x <= v)
        a1_constraints.append(a1 * i <= a0 * (i - 1) * w_h[1, 0] + a1 * (i - 1) * w_h[1, 1])

    plt.title("minimum alpha region")
    draw(a0_constraints + a1_constraints, MIN_GRID, MAX_GRID)


def constraint_conjunction(constraints):
    if isinstance(constraints, list):
        all_constraints = constraints[0]
        for constraint in constraints[1:]:
            all_constraints = all_constraints & constraint
    else:
        all_constraints = constraints
    return all_constraints


def translate_points_to_grid(min_val, max_val, length, points):
    index_point_map = np.array([min_val + ((max_val - min_val) / length * i) for i in range(length)])
    grid_points = []
    for (x, y) in points:
        x_grid = np.argmax(index_point_map > x)
        y_grid = np.argmax(index_point_map > y)
        grid_points.append([x_grid, y_grid])
    return grid_points


def draw(constraints, min_val, max_val, c="Greys", points=None):
    all_constraints = constraint_conjunction(constraints)

    im = plt.imshow((all_constraints).astype(int),
                    extent=(min_val, max_val, min_val, max_val),
                    origin="lower", cmap=c)
    # plt.show()
    if points:
        # grid_points = translate_points_to_grid(min_val, max_val, all_constraints.shape[0], points)
        grid_points = points
        for i in range(0, len(grid_points), 2):
            # x_points = [grid_points[i][0], grid_points[i+1][0]]
            # y_points = [grid_points[i][1], grid_points[i + 1][1]]
            plt.scatter(grid_points[i][0], grid_points[i][1], c='black', s=1)

    plt.xlabel(r'$\alpha_0$')
    plt.ylabel(r'$\alpha_1$')
    plt.show()


def get_multi_d_max_constriants(x, w_in, w_h, num_steps):
    dim = w_h.shape[0]
    r = calc_rnn_values(x, w_in, w_h, num_steps)

    # sample points
    points = np.linspace(MIN_GRID, MAX_GRID, NUM_POINTS_TO_SAMPLE)
    a = np.meshgrid(*[points] * dim)
    # a is a list length dim, each cell is dim dimension array each dimension size NUM_POINTS_TO_SAMPLE
    constraints = []
    for i, cur_alpha in enumerate(a):
        constraints[i] = []
        for j, v in enumerate(r[i, :]):
            # rnn constraint, alpha*i + x >= rnn_value
            constraints[i].append(cur_alpha * j + x * w_in[0] >= v)

            # inductive constraints
            # right_hand_side = 0
            # constraints[i].append(cur_alpha * i >= a0 * (i - 1) * w_h[0, 0] + a1 * (i - 1) * w_h[0, 1])


def get_2d_max_constraints(x, w_in, w_h, num_steps):
    r = calc_rnn_values(x, w_in, w_h, num_steps)
    r0 = r[:, 0]
    r1 = r[:, 1]
    print("r0, r1 values:", list(zip(r0, r1)))
    d = np.linspace(MIN_GRID, MAX_GRID, NUM_POINTS_TO_SAMPLE)
    assert r0[-1] <= MAX_GRID, r0[-1]
    assert r1[-1] <= MAX_GRID, r1[-1]

    a0, a1 = np.meshgrid(d, d)
    induction_constraints = []
    a0_constraints = []
    # Using i and not (i-1) because we start from 1 in r
    for i in range(1, len(r0)):
        v = r0[i]
        # a0_constraints.append(v - ReLU(a0 * i + np.matmul(x, w_in[:, 0])) < EPS)
        # a0_constraints.append(v - ReLU(a0 * i ) < EPS)
        a0_constraints.append(
            v - ReLU(a0 * (i - 1) * w_h[0, 0] + a1 * (i - 1) * w_h[0, 1] + np.matmul(x, w_in[:, 0])) < EPS)
        # we removed + x * w_in[0] from both sides of the equation
        # induction_constraints.append(a0 * i >= ReLU(a0 * (i - 1) * w_h[0, 0] + a1 * (i - 1) * w_h[0, 1] + np.matmul(x, w_in[:, 0])))

    a1_constraints = []
    for i in range(1, len(r1)):
        v = r1[i]
        # a1_constraints.append(ReLU(a1 * i + x) >= v)
        # a1_constraints.append(v - ReLU(a1 * i + np.matmul(x, w_in[:, 1])) < EPS)
        # a1_constraints.append(v - ReLU(a1 * i) < EPS)
        a1_constraints.append(
            v - ReLU(a0 * (i - 1) * w_h[1, 0] + a1 * (i - 1) * w_h[1, 1] + np.matmul(x, w_in[:, 1])) < EPS)
        # a1_constraints.append(np.abs(ReLU(a1 * i + np.matmul(x, w_in[:, 1])) - v) < EPS)
        # we removed + x * w_in[1] from both sides of the equation

    for i in range(1, len(r0)):
        induction_constraints.append(
            a0 * i >= ReLU(a0 * (i - 1) * w_h[0, 0] + a1 * (i - 1) * w_h[0, 1] + np.matmul(x, w_in[:, 0])))
        induction_constraints.append(
            a1 * i >= ReLU(a0 * (i - 1) * w_h[1, 0] + a1 * (i - 1) * w_h[1, 1] + np.matmul(x, w_in[:, 1])))

    max_constraints = a1_constraints + a0_constraints
    # induction_constraints = max_constraints
    return max_constraints, induction_constraints


def get_2d_min_constraints(x, w_in, w_h, num_steps):
    r = calc_rnn_values(x, w_in, w_h, num_steps)
    r0 = r[:, 0]
    r1 = r[:, 1]
    print("r0, r1 values:", list(zip(r0, r1)))
    d = np.linspace(MIN_GRID, MAX_GRID, NUM_POINTS_TO_SAMPLE)
    assert r0[-1] >= MIN_GRID, r0[-1]
    assert r1[-1] >= MIN_GRID, r1[-1]

    a0, a1 = np.meshgrid(d, d)
    induction_constraints = []
    a0_constraints = []
    # Using i and not (i-1) because we start from 1 in r
    for i, v in enumerate(r0):
        a0_constraints.append(ReLU(a0 * i + np.matmul(x, w_in[:, 0]) - v) < EPS)

    a1_constraints = []
    for i, v in enumerate(r1):
        a1_constraints.append(ReLU(a1 * i + np.matmul(x, w_in[:, 1]) - v) < EPS)

    for i in range(1, len(r0)):
        induction_constraints.append(
            a0 * i <= ReLU(a0 * (i - 1) * w_h[0, 0] + a1 * (i - 1) * w_h[0, 1] + np.matmul(x, w_in[:, 0])))
        induction_constraints.append(
            a1 * i <= ReLU(a0 * (i - 1) * w_h[1, 0] + a1 * (i - 1) * w_h[1, 1] + np.matmul(x, w_in[:, 1])))

    min_constraints = a0_constraints + a1_constraints
    return min_constraints, induction_constraints


def draw_min_alphas(x: int, w_in: np.ndarray, w_h: np.ndarray, num_steps: int, points=None):
    min_constraints, induction_constraints = get_2d_min_constraints(x, w_in, w_h, num_steps)
    draw_constraints(min_constraints, induction_constraints, points)


def draw_max_alphas(x: int, w_in: np.ndarray, w_h: np.ndarray, num_steps: int, points=None):
    max_constraints, induction_constraints = get_2d_max_constraints(x, w_in, w_h, num_steps)
    draw_constraints(max_constraints, induction_constraints, points)


def draw_constraints(value_constraints, induction_constraints, points=None):
    # Here we calculate the actual values and check for points that hold them
    plt.title("Actual Values")
    draw(value_constraints, MIN_GRID, MAX_GRID)

    # Here we use the induction hyptoesis which over approximate the actual values
    plt.title("Values that are provable via induction")
    draw(induction_constraints, MIN_GRID, MAX_GRID)

    plt.title("Valid Inductive Invariants")
    draw(value_constraints + induction_constraints, MIN_GRID, MAX_GRID, 'BuPu', points)


def draw_max_constraints(x, w_in, w_h, prop, num_steps):
    max_constraints, induction_constraints = get_2d_max_constraints(x, w_in, w_h, num_steps)
    max_constraints = constraint_conjunction(max_constraints)
    induction_constraints = constraint_conjunction(induction_constraints)
    inductive_overapproximation_constraints = induction_constraints & max_constraints
    d = np.linspace(MIN_GRID, MAX_GRID, NUM_POINTS_TO_SAMPLE)
    a0, a1 = np.meshgrid(d, d)
    property_constraints = (a1 * (num_steps - 1) + a0 * (num_steps - 1) <= prop)

    cmap = 'BuPu'
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
    ax1.set_xlabel('alpha0')
    ax2.set_xlabel('alpha0')
    # ax3.set_xlabel('alpha0')
    ax1.set_ylabel('alpha1')
    # ax2.set_ylabel('alpha1')
    # ax3.set_ylabel('alpha1')

    im_max = np.zeros(max_constraints.shape)
    im_max[max_constraints] = 1
    ax1.imshow(im_max, cmap=cmap, extent=(MIN_GRID, MAX_GRID, MIN_GRID, MAX_GRID), origin="lower", alpha=0.3,
               zorder=1)
    ax1.set_title('Max Constraints')

    im_prop = np.zeros(induction_constraints.shape)
    im_prop[induction_constraints] = 1
    ax2.imshow(im_prop, cmap=cmap, extent=(MIN_GRID, MAX_GRID, MIN_GRID, MAX_GRID), origin="lower", alpha=0.3,
               zorder=1)
    ax2.set_title('Induction Constraints')
    plt.show()

    # im_prop = np.zeros(property_constraints.shape)
    # im_prop[property_constraints] = 1
    # ax2.imshow(im_prop, cmap=cmap, extent=(MIN_GRID, MAX_GRID, MIN_GRID, MAX_GRID), origin="lower", alpha=0.3,
    #            zorder=1)
    # ax2.set_title('Property Implies')
    # plt.show()

    plt.imshow(inductive_overapproximation_constraints, cmap=cmap, extent=(MIN_GRID, MAX_GRID, MIN_GRID, MAX_GRID),
               origin="lower", alpha=0.5,
               zorder=1)
    # plt.imshow(im_induction, cmap=cmap, extent=(MIN_GRID, MAX_GRID, MIN_GRID, MAX_GRID), origin="lower", alpha=0.4,
    #            zorder=1)
    plt.imshow(im_prop, cmap=cmap, extent=(MIN_GRID, MAX_GRID, MIN_GRID, MAX_GRID), origin="lower", alpha=0.3,
               zorder=1)
    plt.title("Alphas for proving property <= {}".format(prop))
    plt.xlabel('alpha0')
    plt.ylabel('alpha1')
    plt.show()


def func_to_gif(gif_name, func, num_steps, args, start_time_stamp=2):
    gif_name.replace("\n", "\\n")
    gif_dir_name, file_name = gif_name.split('/')
    images_dir_name = os.path.join(gif_dir_name, "images_" + file_name)
    os.makedirs(images_dir_name)

    for i in tqdm(range(start_time_stamp, num_steps + 1)):
        func(**args, num_steps=i, file_name="{}/{}".format(images_dir_name, i))

    import imageio
    images = []
    for filename in range(start_time_stamp, num_steps + 1):
        images.append(imageio.imread("{}/{}.png".format(images_dir_name, filename)))
    imageio.mimsave('{}.gif'.format(gif_name), images, duration=0.5)


def draw_max_alphas_and_property(x: int, w_in: np.ndarray, w_h: np.ndarray, property: int, num_steps: int,
                                 file_name: str = None):
    '''
    the
    :param x:
    :param w_in:
    :param w_h:
    :param property: int that we want r0 + r1 to be smaller then
    :param num_steps:
    :return:
    '''
    a0_constraints, a1_constraints = get_2d_max_constraints(x, w_in, w_h, num_steps)
    constraints = a0_constraints + a1_constraints
    valid_invariants = constraints[0]
    for constraint in constraints[1:]:
        valid_invariants = valid_invariants & constraint

    d = np.linspace(MIN_GRID, MAX_GRID, NUM_POINTS_TO_SAMPLE)
    a0, a1 = np.meshgrid(d, d)

    values = [100, 50, 0]
    property_constraints = (a1 * (num_steps - 1) + a0 * (num_steps - 1) <= property)
    property_and_invariant = np.zeros(property_constraints.shape) + 200
    # property_and_invariant[property_constraints] = values[0]
    # property_and_invariant[valid_invariants] = values[1]
    # property_and_invariant[property_constraints & valid_invariants] = values[2]

    im = plt.imshow((property_and_invariant).astype(int),
                    extent=(MIN_GRID, MAX_GRID, MIN_GRID, MAX_GRID),
                    origin="lower", cmap='hot')
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[0], label="Property Holds"),
               mpatches.Patch(color=colors[1], label="Invariant Holds"),
               mpatches.Patch(color=colors[2], label="Property and Invariant Holds")
               ]
    # put those patched as legend-handles into the legend
    plt.title("Alphas for proving max property, n: {} w_h: {}".format(num_steps, w_h))
    plt.legend(handles=patches, loc=2)
    plt.xlabel('alpha0')
    plt.ylabel('alpha1')
    plt.show()
    # plt.savefig(file_name)
    # draw(property_and_invariant, MIN_GRID, MAX_GRID)


def draw_2d_hidden_output_from_h5(x, h5_path, num_steps):
    model = keras.models.load_model(h5_path)
    w_in, w_h, b = model.layers[0].get_weights()
    draw_2d_hidden_output_one_image(x, w_in, w_h, num_steps)


def draw_2d_hidden_output_one_image(x, w_in, w_h, num_steps):
    r = calc_rnn_values(x, w_in, w_h, num_steps)
    print(r)
    colors = ['r', 'g', 'b']
    for i in range(r.shape[1]):
        plt.scatter(range(len(r[:, i])), r[:, i], label='r' + str(i), color=colors[i], alpha=0.4)
    def draw_line(m, b, i, is_upper):
        label_text = 'upper' if is_upper else 'lower'
        linestyle = 'dashed' if is_upper else 'dotted'
        plt.plot(line_range, [m * x + b for x in line_range], label='r{}_{}'.format(i, label_text),
                 color=colors[i], linestyle=linestyle, alpha=0.6)

    line_range = range(-1, r.shape[0])
    for i in range(r.shape[1]):
        continue
        if i == 0:
            draw_line(m=0, b=0, i=i, is_upper=False)
            draw_line(m=0.495, b=1, i=i, is_upper=True)
        if i == 1:
            draw_line(m=0, b=0, i=i, is_upper=False)
            draw_line(m=0.912, b=1, i=i, is_upper=True)


    plt.title("RNN Output")
    plt.xlabel('time')
    plt.ylabel('rnn_out')
    plt.legend(loc='best')
    # plt.savefig(file_name)
    plt.show()


def draw_4d_hidden_values(x, w_in, w_h0, num_steps):
    return
    from tqdm import tqdm

    min_r = 0
    max_r = 1.5
    points_to_sample = 5 ** 2
    sqrt_num_points = int(np.sqrt(points_to_sample))
    # valid = np.zeros((sqrt_num_points, sqrt_num_points, sqrt_num_points, sqrt_num_points))
    step_size = (max_r - min_r) / sqrt_num_points

    pbar = tqdm(total=points_to_sample * num_steps)
    invalid_points = None  # np.array([])
    c = 0
    for t in range(1, num_steps + 1):
        for i in range(sqrt_num_points):
            for j in range(sqrt_num_points):
                # print(t, i * step_size, j * step_size)
                # c += 1
                # continue
                # for k in range(sqrt_num_points):
                #     for l in range(sqrt_num_points):
                w_h = np.vstack(
                    (np.array([i * step_size, j * step_size]), np.array([1, 1])))  # [k * step_size, l * step_size])))
                a0_constraints, a1_constraints = get_max_constriants(x, w_in, w_h, t)
                # valid[i,j, k, l] = 1 if constraint_conjunction(a0_constraints + a1_constraints).any() else 0
                if not constraint_conjunction(a0_constraints + a1_constraints).any():
                    if invalid_points is not None:
                        valid[i, j, k, l] = 1 if constraint_conjunction(a0_constraints + a1_constraints).any() else 0
                        invalid_points = np.vstack(
                            (invalid_points, [t, i * step_size, j * step_size]))  # , k*step_size, l*step_size]))
                    else:
                        invalid_points = [t, i * step_size, j * step_size]
                pbar.update(1)
                # if valid[i,j]:
                #     pass
                # print("\n", i*step_size, j*step_size, k*step_size, l*step_size)
    print(c)
    print("\n", invalid_points)
    print("max values:", i * step_size, j * step_size)  # , k*step_size, l*step_size)

    # im = plt.imshow((valid).astype(int),
    #                 extent=(0, 1, 0, 1),
    #                 origin="lower", cmap='Greys')
    # plt.title("Valid values for W_h = [ [{}, {}], x, y]".format(w_h0[0], w_h0[1]))
    # plt.xlabel('w_h_1_0')
    # plt.ylabel('w_h_1_1')
    # plt.show()


def draw_2d_from_h5(h5_path, in_tensor, steps, algorithm_points=None):
    '''
    Draws 2d limits by h5 file
    :param h5_path: path to h5 with rnn model
    :param in_tensor: in tensor for the model to evaluate the regions
    :param steps: number of steps (time)
    :param algorithm_points: None, or a list of points, each point 2d tupple to add a search pattern in the drawing
    :return:
    '''
    from RNN.MarabouRnnModel import RnnMarabouModel
    rnnModel = RnnMarabouModel(h5_path, steps)
    w_in, w_h, b = rnnModel.get_weights()[0]
    draw_max_alphas(in_tensor, w_in, w_h, num_steps=steps, points=algorithm_points)


if __name__ == "__main__":
    # in_tensor = np.array([0.23300637, 0.0577466 , 0.88960908, 0.02926062, 0.4322654 ,
    #     0.05116153, 0.93342266, 0.3143915 , 0.39245229, 0.1144419 ,
    #     0.08748452, 0.24332963, 0.34622415, 0.42573235, 0.26952168,
    #     0.53801347, 0.26718764, 0.24274057, 0.11475819, 0.9423371 ,
    #     0.70257952, 0.34443971, 0.08917664, 0.50140514, 0.75890139,
    #     0.65532994, 0.74165648, 0.46543468, 0.00583174, 0.54016713,
    #     0.74460554, 0.45771724, 0.59844178, 0.73369685, 0.50576504,
    #     0.91561612, 0.39746448, 0.14791963, 0.38114261, 0.24696231])
    # draw_2d_from_h5("models/model_classes5_1rnn2_0_64_4.h5", in_tensor, 5)
    # exit(0)
    steps = 4
    max_property = 15

    x = np.array([1])
    draw_2d_hidden_output_from_h5(x, 'FMCAD_EXP/rnn_test_model.h5', steps)
    exit(1)

    w_in = np.array([1, 1])[None, :]

    # w_h = np.array([[0.1, 0.7], [1, 1]]) # no convex shape
    # w_h0 = [0.5,0.5]
    # w_h0 = [-0.2, 2]
    w_h = np.array([[1, 1], [1, -1]])

    draw_2d_hidden_output_one_image(x, w_in, w_h, steps)
    draw_max_constraints(x, w_in, w_h, 10, 2)
    # draw_min_alphas(x, w_in, w_h, steps)

    # draw_max_alphas_and_property(x, w_in, w_h, max_property, steps)
    # draw_min_alphas(x, w_in, w_h, steps)

    # r0, r1 = calc_rnn_values(1, np.array([1,1]), np.array([[1,1], [0, 0]]), 10)
    # draw_r_values(r0, r1, 0.1)
