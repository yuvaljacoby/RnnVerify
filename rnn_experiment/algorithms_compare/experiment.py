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
from RNN.MultiLayerBase import GurobiMultiLayer
from rns_verify.verify_keras import verify_query as rns_verify_query

# BASE_FOLDER = "/home/yuval/projects/Marabou/"
BASE_FOLDER = "."
CS_BASE_FOLDER = "/cs/usr/yuvalja/projects/Marabou"
if os.path.exists(CS_BASE_FOLDER):
    BASE_FOLDER = CS_BASE_FOLDER

MODELS_FOLDER = os.path.join(BASE_FOLDER, "models")
FIGUERS_FOLDER = os.path.join(BASE_FOLDER, "figures")
POINTS_PICKLE = os.path.join(MODELS_FOLDER, 'points.pkl')
PICKLE_DIR = os.path.join(BASE_FOLDER, "pickles/rns_verify_exp")
os.makedirs(PICKLE_DIR, exist_ok=True)

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
        if os.path.isfile(sys.argv[2]):
            parse_results_file(sys.argv[2])
        else:
            parse_results_file_per_time(sys.argv[2])
    if sys.argv[1] == 'exp':
        max_iterations = MAX_ITERATIONS
        if len(sys.argv) > 2:
            max_iterations = int(sys.argv[2]) + 1 # Make max_iterations inclusive
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
    assert all([l == len_results[0] for l in len_results])

    avg = lambda x: sum(x) / len(x) if len(x) > 0 else 0
    our_avg = [-1] * len(results.keys())
    rns_avg = [-1] * len(results.keys())
    start_idx = min(results.keys())
    for k, v in results.items():
        our_avg[k - start_idx] = avg(v['our'])
        rns_avg[k - start_idx] = avg(v['rns'])

    plot_results(our_avg, rns_avg, "Compare RNSVerify and RNNVerifiy on {} points".format(len_results[0]))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse_inputs()
        exit(0)
    print("USAGE is one of the following:")
    print("\texp <T_max> --> compares RnnVerify and RNSVerify for every 2 <= i <= T_max")
    print("\texact <T> --> compares RnnVerify and RNSVerify for the given T")
    print("\tanalyze <path> --> Prints results from a single pickle or combining multiple pickles from the same folder")

