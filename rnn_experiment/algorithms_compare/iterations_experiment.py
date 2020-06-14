import os
import pickle
import sys
import time
from functools import partial
from timeit import default_timer as timer
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from RNN.Adversarial import adversarial_query, get_out_idx
from polyhedron_algorithms.GurobiBased.MultiLayerBase import GurobiMultiLayer
from rns_verify.verify_keras import verify_query as rns_verify_query

# BASE_FOLDER = "/home/yuval/projects/Marabou/"
BASE_FOLDER = "../.."
CS_BASE_FOLDER = "/cs/usr/yuvalja/projects/Marabou"
if os.path.exists(CS_BASE_FOLDER):
    BASE_FOLDER = CS_BASE_FOLDER

MODELS_FOLDER = os.path.join(BASE_FOLDER, "models")
FIGUERS_FOLDER = os.path.join(BASE_FOLDER, "figures")
POINTS_PICKLE = os.path.join(BASE_FOLDER, "pickles", 'points.pkl')
PICKLE_DIR = os.path.join(BASE_FOLDER, "pickles/rns_verify_exp")

NUM_POINTS = 25
MAX_ITERATIONS = 25
DEFAULT_H5 = os.path.join(MODELS_FOLDER, 'model_marabou_rnsverify_compare.h5')


def run_exp_signle_time(points, radius, h5_file, t, only_rns=False, pbar=None, save_results=False):
    our_raw, rns_raw = [], []
    for j, point in enumerate(points):
        idx_max, other_idx = get_out_idx(point, t, h5_file, lambda x: np.argsort(x)[-2])
        # idx_max, other_idx = None, None
        rnsverify_time = -1
        rnsverify_time = rns_verify_query(h5_file, point, idx_max, other_idx, t, radius)

        our_time = -1
        if not only_rns:
            gurobi_ptr = partial(GurobiMultiLayer, use_relu=True, add_alpha_constraint=True,
                                 use_counter_example=True)
            try:
                start = timer()
                res, _, _ = adversarial_query(point, radius, idx_max, other_idx, h5_file, gurobi_ptr, t)
                our_time = timer() - start
            except ValueError:
                res = False
                our_time = -1
            assert res
        # total_ours += our_time
        # total_rns += rnsverify_time
        if pbar:
            pbar.update(1)

        our_raw.append(our_time)
        rns_raw.append(rnsverify_time)
        print('time: {}, point: {} our: {}, rns: {}'.format(t, j, our_time, rnsverify_time))

    if save_results:
        exp_name = 'verification time as a function of iterations, one rnn cell over {} points, time: {}'.format(
            len(points), t)
        file_name = "rns_{}time{}_{}.pkl".format('' if only_rns else 'ours_', t, time.strftime("%Y%m%d-%H%M%S"))
        pickle_path = os.path.join(PICKLE_DIR, file_name)
        print("#" * 100)
        print(" " * 20 + "PICKLE PATH: {}".format(pickle_path))
        print("#" * 100)
        pickle.dump({'our_raw': our_raw, 'rns_raw': rns_raw, 'exp_name': exp_name}, open(pickle_path, "wb"))

    return our_raw, rns_raw


def run_multiple_experiment(points, radius, h5_file, max_iterations=100, only_rns=False):
    our_results = []
    rnsverify_results = []
    our_raw, rns_raw = [], []

    avg = lambda x: sum(x) / len(x) if len(x) > 0 else 0
    pbar = tqdm(total=len(range(2, max_iterations)) * len(points))
    for i in range(2, max_iterations):
        our_p, rns_p = run_exp_signle_time(points, radius, h5_file, i, only_rns, pbar)

        our_results.append(avg(our_p))
        rnsverify_results.append(avg(rns_p))
        our_raw.append(our_p)
        rns_raw.append(rns_p)

    exp_name = 'verification time as a function of iterations, one rnn cell over {} points'.format(len(points))
    file_name = "{}_{}_{}.pkl".format(h5_file.split('/')[-1].split(".")[-2], max_iterations,
                                      time.strftime("%Y%m%d-%H%M%S"))
    pickle_path = os.path.join(PICKLE_DIR, file_name)
    print("#" * 100)
    print(" " * 20 + "PICKLE PATH: {}".format(pickle_path))
    print("#" * 100)
    pickle.dump(
        {'our': our_results, 'rns': rnsverify_results, 'our_raw': our_raw, 'rns_raw': rns_raw, 'exp_name': exp_name},
        open(pickle_path, "wb"))
    print_table(our_results, rnsverify_results)


def run_experiment(in_tensor, radius, idx_max, other_idx, h5_file, max_iterations=100):
    return run_multiple_experiment([in_tensor], radius, idx_max, other_idx, h5_file, max_iterations)


def print_table(our_results: List[float], rnsverify_results: List[float]):
    from prettytable import PrettyTable
    x = PrettyTable()
    x.field_names = ['Tmax'] + ['our', 'rns_verify']
    for i, (our, rns) in enumerate(zip(our_results, rnsverify_results)):
        x.add_row([i + 2, our, rns])
    print(x)


def plot_results(our_results, rnsverify_results, exp_name):
    assert len(our_results) == len(rnsverify_results)
    print_table(our_results, rnsverify_results)
    x_idx = range(2, len(our_results) + 2)

    plt.figure(figsize=(13, 9))
    dot_size = 800
    sns.scatterplot(x_idx, our_results, s=dot_size)
    sns.scatterplot(x_idx, rnsverify_results, s=dot_size)

    plt.legend(['RnnVerify', 'RNSVerify'], loc='upper left', fontsize=32)

    plt.xlabel('Number of Iterations ($T_{max}$)', fontsize=36)
    plt.ylabel('Time (seconds)', fontsize=36)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.savefig((FIGUERS_FOLDER + "rns_ours_rnn2_fc0_avg.pdf").replace(' ', '_'), dpi=100)
    plt.show()


def parse_results_file(path):
    if not os.path.exists(path):
        pickle_dir = "pickles/rns_verify_exp/"
        path = os.path.join(pickle_dir, path)
        if not os.path.exists():
            raise FileNotFoundError

    data = pickle.load(open(path, "rb"))
    our = data['our']
    rns = data['rns']
    exp_name = data['exp_name']
    # plot_results(our, rns, exp_name)
    print_table(our, rns)


def parse_inputs():
    points = pickle.load(open(POINTS_PICKLE, "rb"))[:NUM_POINTS]
    r = 0.01
    if sys.argv[1] == 'analyze':
        parse_results_file_per_time(sys.argv[2])
    if sys.argv[1] == 'exp':
        max_iterations = MAX_ITERATIONS
        if len(sys.argv) > 2:
            max_iterations = int(sys.argv[2])
        run_multiple_experiment(points, r, DEFAULT_H5, max_iterations)
    if sys.argv[1] == 'exact':
        t = int(sys.argv[2])
        run_exp_signle_time(points, r, DEFAULT_H5, t, save_results=1, only_rns=0)

def parse_results_file_per_time(dir_path: str):
    if not os.path.exists(dir_path):
        raise FileNotFoundError
    results = {}
    len_results = []
    for f in os.listdir(dir_path):
        if not "rns_ours_time" in f:
            continue
        data = pickle.load(open(os.path.join(dir_path, f), "rb"))
        t = f.replace("rns_ours_time", "").split("_")[0]
        results[int(t)] = {'our': data['our_raw'], 'rns': data['rns_raw']}
        assert len(data['our_raw']) == len(data['rns_raw'])
        len_results.append(len(data['our_raw']))
    print([l for l in len_results])
    assert all([l == len_results[0] for l in len_results])

    avg = lambda x: sum(x) / len(x) if len(x) > 0 else 0
    our_avg = [-1] * len(results.keys())
    rns_avg = [-1] * len(results.keys())
    start_idx = min(results.keys())
    for k, v in results.items():
        our_avg[k - start_idx] = avg(v['our'])
        rns_avg[k - start_idx] = avg(v['rns'])

    plot_results(our_avg, rns_avg, "Compare RNSVerify and RNNVerifiy on {} points".format(len_results[0]))
    print_table(our_avg, rns_avg)


experiemnts = [
    # {'idx_max': 9, 'other_idx': 2,
    #  'in_tensor': [10] * 40, 'radius': 0.01,
    #  'h5_path': "{}/model_classes20_1rnn2_0_64_4.h5".format(MODELS_FOLDER), 'n_iterations': 25},
    # {'idx_max': None, 'other_idx': None,
    #  'in_tensor': np.array([2.21710942e-03, -5.79088139e-01, -2.23213261e+00, -2.57655135e-02,
    #                         -7.56722928e-01, -9.62270726e-01, -3.03466236e+00, -9.81743962e-01,
    #                         -4.81361157e-01, -1.29589492e+00, 1.27178216e+00, 3.48023461e+00,
    #                         5.93364435e-01, 1.41500732e+00, 3.64563153e+00, 8.61538059e-01,
    #                         3.08545925e+00, -1.80144234e+00, -2.74250021e-01, 2.59515802e+00,
    #                         1.35054233e+00, 6.39162339e-02, 1.83629179e+00, 7.61018933e-01,
    #                         1.03273497e+00, -7.10478917e-01, 4.17554002e-01, 6.56822152e-01,
    #                         -9.96449533e-01, -4.18355355e+00, -1.65175481e-01, 4.91036530e+00,
    #                         -5.34422001e+00, -1.82655856e+00, -4.54628714e-01, 5.38630754e-01,
    #                         2.26092251e+00, 2.08479489e+00, 2.60762089e+00, 2.77880146e+00]), 'radius': 0.01,
    #  'h5_path': "ATVA_EXP/models/epochs100/model_20classes_rnn2_fc32_fc32_fc32_fc32_fc32_epochs100.h5"
    #     , 'n_iterations': 25},
    # {'idx_max': None, 'other_idx': None,
    #  'in_tensor': np.array([2.21710942e-03, -5.79088139e-01, -2.23213261e+00, -2.57655135e-02,
    #    -7.56722928e-01, -9.62270726e-01, -3.03466236e+00, -9.81743962e-01,
    #    -4.81361157e-01, -1.29589492e+00,  1.27178216e+00,  3.48023461e+00,
    #     5.93364435e-01,  1.41500732e+00,  3.64563153e+00,  8.61538059e-01,
    #     3.08545925e+00, -1.80144234e+00, -2.74250021e-01,  2.59515802e+00,
    #     1.35054233e+00,  6.39162339e-02,  1.83629179e+00,  7.61018933e-01,
    #     1.03273497e+00, -7.10478917e-01,  4.17554002e-01,  6.56822152e-01,
    #    -9.96449533e-01, -4.18355355e+00, -1.65175481e-01,  4.91036530e+00,
    #    -5.34422001e+00, -1.82655856e+00, -4.54628714e-01,  5.38630754e-01,
    #     2.26092251e+00,  2.08479489e+00,  2.60762089e+00,  2.77880146e+00]), 'radius': 0.01,
    #  'h5_path': "{}/model_20classes_rnn4_fc32_fc32_fc32_fc32_fc32_epochs50.h5".format(MODELS_FOLDER), 'n_iterations': 25},
    {'idx_max': 19, 'other_idx': 8,
     'in_tensor': np.array([2.21710942e-03, -5.79088139e-01, -2.23213261e+00, -2.57655135e-02,
                            -7.56722928e-01, -9.62270726e-01, -3.03466236e+00, -9.81743962e-01,
                            -4.81361157e-01, -1.29589492e+00, 1.27178216e+00, 3.48023461e+00,
                            5.93364435e-01, 1.41500732e+00, 3.64563153e+00, 8.61538059e-01,
                            3.08545925e+00, -1.80144234e+00, -2.74250021e-01, 2.59515802e+00,
                            1.35054233e+00, 6.39162339e-02, 1.83629179e+00, 7.61018933e-01,
                            1.03273497e+00, -7.10478917e-01, 4.17554002e-01, 6.56822152e-01,
                            -9.96449533e-01, -4.18355355e+00, -1.65175481e-01, 4.91036530e+00,
                            -5.34422001e+00, -1.82655856e+00, -4.54628714e-01, 5.38630754e-01,
                            2.26092251e+00, 2.08479489e+00, 2.60762089e+00, 2.77880146e+00]), 'radius': 0.01,
     'h5_path': "{}/model_marabou_rnsverify_compare.h5".format(MODELS_FOLDER), 'n_iterations': 25},
]

if __name__ == "__main__":
    # idx_max = 4
    # other_idx = 0
    # in_tensor = [10] * 40
    # n_iterations = 20  # 1000?
    # r = 0
    # model_path = 'models/model_classes5_1rnn2_0_64_4.h5'
    # results_path = "pickles/rns_verify_exp/model_marabou_rnsverify_compare_25_20200504-004813.pkl"
    # d = pickle.load(open(results_path, "rb"))
    # plot_results(d['our'], d['rns'], d['exp_name'])
    # exit(0)
    parse_results_file_per_time("pickles/rns_verify_exp/")
    exit(0)
    if len(sys.argv) > 1:
        parse_inputs()
        exit(0)

    for exp in experiemnts:
        if exp['idx_max'] is None:
            exp['idx_max'], exp['other_idx'] = get_out_idx(exp['in_tensor'], exp['n_iterations'], exp['h5_path'])
            print(exp['idx_max'], exp['other_idx'])
        our, rns = run_experiment(exp['in_tensor'], exp['radius'], exp['idx_max'], exp['other_idx'],
                                  exp['h5_path'], max_iterations=exp['n_iterations'])
        rnn_dim = exp['h5_path'].split('/')[-1].split('_')[2].replace('1rnn', '')
        exp_name = 'verification time as a function of iterations, one rnn cell dimension: {}'.format(rnn_dim)

        pickle_path = PICKLE_DIR + "{}_{}_{}.pkl".format(exp['h5_path'].split("/")[-1].split(".")[-2],
                                                         exp['n_iterations'], time.strftime("%Y%m%d-%H%M%S"))
        print("#" * 100)
        print(" " * 20 + "PICKLE PATH: {}".format(pickle_path))
        print("#" * 100)
        # pickle.dump({'our': our, 'rns': rns, 'exp_name': exp_name}, open(pickle_path, "wb"))

        # plot_results(our, rns, exp_name)
        print_table(our, rns)
