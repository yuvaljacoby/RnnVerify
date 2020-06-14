# BASE_FOLDER = "/cs/usr/yuvalja/projects/Marabou"
# import sys
#
# sys.path.insert(0, BASE_FOLDER)

import os
import pickle
import sys
import traceback
from collections import OrderedDict
from datetime import datetime
from functools import partial
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from tqdm import tqdm

# TODO: Recall what we did how to run and move to polyhedron algorithms, and RNN folder
from RNN.Adversarial import adversarial_query, get_out_idx
from polyhedron_algorithms.GurobiBased.GurobiPolyhedronIISBased import GurobiMultiLayerIIS
from polyhedron_algorithms.GurobiBased.GurobiPolyhedronRandomImprove import GurobiMultiLayerRandom
from polyhedron_algorithms.GurobiBased.MultiLayerBase import GurobiMultiLayer
from rnn_experiment.self_compare.create_sbatch_iterations_exp import BASE_FOLDER

# from rnn_algorithms.GurobiBased import AlphasGurobiBased
# from rnn_algorithms.IterateAlphasSGD import IterateAlphasSGD
# from rnn_algorithms.RandomAlphasSGD import RandomAlphasSGD
# from rnn_algorithms.Update_Strategy import Absolute_Step, Relative_Step

MODELS_FOLDER = os.path.join(BASE_FOLDER, "FMCAD_EXP/models/")
EXPERIMENTS_FOLDER = os.path.join(BASE_FOLDER, "working_arrays/")
IN_SHAPE = (40,)
SBATCH_FOLDER = "sbatch_exp"

DEBUG = 1


def run_one_comparison(in_tensor, radius, idx_max, other_idx, h5_file, n_iterations, algorithms_ptrs, steps_num=2500,
                       return_dict=False):
    results = {}

    for name, algo_ptr in algorithms_ptrs.items():
        print('starting algo:' + name)
        start = timer()

        try:
            res, queries_stats, alpha_history = adversarial_query(in_tensor, radius, idx_max, other_idx, h5_file,
                                                                  algo_ptr, n_iterations, steps_num)
        except ValueError as e:
            # row_result = {'point': in_tensor, 'error': e, 'error_traceback': traceback.format_exc(), 'result' : False}
            res = False
            queries_stats = {}
            queries_stats['invariant_queries'] = -1
            queries_stats['property_queries'] = -1
            queries_stats['invariant_times'] = []
            queries_stats['property_times'] = []
            queries_stats['number_of_updates'] = -1
        # res = False
        # iterations = 23
        end = timer()
        if queries_stats is None:
            return None
        results[name] = {'time': end - start, 'result': res,
                         'invariant_iterations': queries_stats['invariant_queries'],
                         'property_iterations': queries_stats['property_queries'],
                         'invariant_times': queries_stats['invariant_times'],
                         'property_times': queries_stats['property_times'],
                         'iterations': queries_stats['number_of_updates'],
                         'in_tensor': in_tensor
                         }
        print("%%%%%%%%% {} {} %%%%%%%%%".format(res, end - start))

    if return_dict:
        return results
    # print(results)
    row_result = [results[n]['result'] for n in algorithms_ptrs.keys()] + \
                 [results[n]['iterations'] for n in algorithms_ptrs.keys()] + \
                 [results[n]['time'] for n in algorithms_ptrs.keys()] + \
                 [results[n]['invariant_iterations'] for n in algorithms_ptrs.keys()] + \
                 [results[n]['property_iterations'] for n in algorithms_ptrs.keys()] + \
                 [results[n]['invariant_times'] for n in algorithms_ptrs.keys()] + \
                 [results[n]['property_times'] for n in algorithms_ptrs.keys()] + \
                 [results[n]['in_tensor'] for n in algorithms_ptrs.keys()]

    return row_result


def get_random_input(model_path, mean, var, n_iterations):
    #     return [14.27122768, 10.01429519, 15.79244755, 12.31632729, 10.77446205, 11.6998685
    # , 10.02309098, 10.20288622, 10.1177965, 12.98410927, 11.80191447, 7.18403711
    # , 8.64965939, 12.03823125, 11.88635415, 14.10741009, 12.95682361, 7.72710876
    # , 10.92425513, 5.54067457, 7.44212401, 14.02778702, 4.92153551, 6.81608973
    # , 7.61313801, 10.73096574, 15.37313871, 5.519518, 8.77897563, 12.87859216
    # , 11.5303272, 5.33148159, 9.86905325, 5.87211545, 5.12149148, 8.93463704
    # , 7.61022822, 5.07853598, 17.71975044, 8.69002408], 14, 11

    while True:
        in_tensor = np.random.normal(mean, var, IN_SHAPE)

        # if any(in_tensor < 0):
        #     print("resample got negative input")
        #     continue
        y_idx_max, other_idx = get_out_idx(in_tensor, n_iterations, model_path)
        if y_idx_max is not None and other_idx is not None and y_idx_max != other_idx:
            # in_tensor = np.array(
            #     [-1.3047684, 5.46331191, 1.93008573, 5.54210032, -0.04579439, 0.84698066, 0.88733042, 0.36111682,
            #      -0.89590958, 2.02979288, 0.02477424, 1.50918829, 1.8345788, 2.26410531, 3.49979787, 0.42402515,
            #      -1.22385631, 0.78972247, -0.18285229, -1.71556589, -0.34333373, -0.4077247, -1.32055327, 3.3423448,
            #      0.20721657, -2.58905041, 4.83447012, -0.25091597, 1.27664352, 2.0043919, -3.37314246, 2.1957612,
            #      -2.1478245, 1.44939961, 1.59584935, 2.38236111, -1.84593505, 1.24174073, 2.45039407, 1.94192])
            # y_idx_max = 14
            # other_idx = 11
            print(in_tensor)
            return in_tensor, y_idx_max, other_idx


def run_controlled_experiment(model_name, algorithms_ptrs, points, other_idx_method, radius=0.01, n_iterations=5, start_idx=0,
                              steps_num=1500):
    '''
    Run a controled experiment, using the given points, start_idx will indicate from where to start iterating the points
    The return value is dicitionary of raw_results, and the key is the index in points
    Other then that there is a "experiment_details" key with other data (such as radius, n_iterations etc)
    '''
    model_path = model_name
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_FOLDER, model_name)

    pickle_path = model_name.split('/')[-1] + str(radius) + "_".join(algorithms_ptrs.keys()) + \
                  str(datetime.now()).replace('.', '').replace(' ', '')
    print("###############results in pickle:\n{}\n#############################################".format(pickle_path))
    results = {'experiment_details': {'radius': radius, 'n_iterations': n_iterations, 'start_idx': start_idx,
                                      'model_name': model_name, 'start_time': datetime.now()}}
    for method in other_idx_method:
        for i, point in tqdm(enumerate(points[start_idx:])):
            y_idx_max, other_idx = get_out_idx(point, n_iterations, model_path, method)
            try:
                row_result = run_one_comparison(point, radius, y_idx_max, other_idx, model_path, n_iterations,
                                                algorithms_ptrs, steps_num=steps_num, return_dict=True)
            except Exception as e:
                if DEBUG:
                    raise e
                row_result = {'point': point, 'error': e, 'error_traceback': traceback.format_exc(), 'result': False}
            results.update({i + start_idx: row_result})
            pickle.dump(results, open("controlled_{}.pkl".format(pickle_path), "wb"))
    return results


def run_random_experiment(model_name, algorithms_ptrs, num_points=150, mean=10, var=3, radius=0.01, n_iterations=50,
                          steps_num=1000):
    '''
    runs comperasion between all the given algorithms on num_points each pointed sampled from Normal(mean,var)
    :param model_name: h5 file in MODELS_FOLDER
    :return: DataFrame with results
    '''
    cols = ['exp_name'] + ['{}_result'.format(n) for n in algorithms_ptrs.keys()] + \
           ['{}_queries'.format(n) for n in algorithms_ptrs.keys()] + \
           ['{}_time'.format(n) for n in algorithms_ptrs.keys()] + \
           ['{}_invariant_queries'.format(n) for n in algorithms_ptrs.keys()] + \
           ['{}_property_queries'.format(n) for n in algorithms_ptrs.keys()] + \
           ['{}_invariant_times'.format(n) for n in algorithms_ptrs.keys()] + \
           ['{}_property_times'.format(n) for n in algorithms_ptrs.keys()] + \
           ['{}_in_tensor'.format(n) for n in algorithms_ptrs.keys()]

    df = pd.DataFrame(columns=cols)
    model_path = os.path.join(MODELS_FOLDER, model_name)
    # pickle_path = model_name + "_randomexp_" + "_".join(algorithms_ptrs.keys()) + str(datetime.now()).replace('.', '')
    pickle_path = model_name + str(radius) + "_".join(algorithms_ptrs.keys()) + \
                  str(datetime.now()).replace('.', '').replace(' ', '')
    for _ in tqdm(range(num_points)):
        in_tensor, y_idx_max, other_idx = get_random_input(model_path, mean, var, n_iterations)

        row_result = run_one_comparison(in_tensor, radius, y_idx_max, other_idx,
                                        model_path,
                                        n_iterations, algorithms_ptrs, steps_num=steps_num)
        if row_result is None:
            print("Got out vector with all entries equal")
            continue
        exp_name = model_path.split('.')[0].split('/')[-1] + '_' + str(n_iterations)
        df = df.append({cols[i]: ([exp_name] + row_result)[i] for i in range(len(row_result) + 1)}, ignore_index=True)
        print(df[[n for n in df.columns if "result" in n or 'queries' in n]])
        # pickle.dump(df, open("results_{}.pkl".format(pickle_path), "wb"))
    return df


def run_experiment_from_pickle(pickle_name, algorithms_ptrs):
    '''
    The search_for_input method is creating a pickle with all the examples, read that and compare algorithms using the
    examples from there
    :param pickle_name: name of file inside the EXPERIMENTS_FOLDER
    :param algorithms_ptrs: pointers to algorithms to run the experiment on
    :return: DataFrame with experiment results
    '''
    pickle_path = os.path.join(EXPERIMENTS_FOLDER, pickle_name)
    experiemnts = pickle.load(open(pickle_path, "rb"))
    model_name = pickle_name.replace(".pkl", "")
    model_path = "{}/{}.h5".format(MODELS_FOLDER, model_name)
    cols = ['exp_name'] + ['{}_result'.format(n) for n in algorithms_ptrs.keys()] + \
           ['{}_queries'.format(n) for n in algorithms_ptrs.keys()] + \
           ['{}_time'.format(n) for n in algorithms_ptrs.keys()]
    df = pd.DataFrame(columns=cols)

    for exp in experiemnts:
        row_result = run_one_comparison(exp['in_tensor'], exp['radius'], exp['idx_max'], exp['other_idx'],
                                        model_path,
                                        exp['n_iterations'], algorithms_ptrs)
        exp_name = model_path.split('.')[0].split('/')[-1] + '_' + str(exp['n_iterations'])
        df = df.append({cols[i]: ([exp_name] + row_result)[i] for i in range(len(row_result) + 1)}, ignore_index=True)
        print(df)
        pickle_path = model_name + "_".join(algorithms_ptrs.keys())
        pickle.dump(df, open("results_{}.pkl".format(pickle_path), "wb"))

    return df


# def get_algorithms():
#     Absolute_Step_Big = partial(Absolute_Step, options=[10 ** i for i in range(-5, 3)])
#     Absolute_Step_Fixed = partial(Absolute_Step, options=[0.1])
#     Relative_Step_Fixed = partial(Absolute_Step, options=[0.05])
#     Relative_Step_Big = partial(Absolute_Step, options=[0.01, 0.05, 0.1, 0.3])
#     sigmoid = lambda x: 1 / (1 + np.exp(-x))
#
#     def create_gurobi_permutations(compare_entry):
#         possible_values = {
#             'update_strategy_ptr': [Relative_Step],  # Absolute_Step,
#             'random_threshold': [20, 1000],  # [5, 20,100,1000],
#             'use_relu': [True, False],
#             'add_alpha_constraint': [True, False],
#             'use_counter_example': [True, False],
#         }
#         from itertools import product
#         cartesian_product = [OrderedDict(zip(possible_values, v)) for v in product(*possible_values.values())]
#         experiments = {}
#         for entry in cartesian_product:
#             entry_name = "gurobi_" + "_".join([str(v) for v in entry.values() if type(v) != type])
#             entry_pointer = partial(AlphasGurobiBased, **entry)
#             experiments.update({'{}_{}'.format(entry_name, compare_entry[0]): OrderedDict({
#                 entry_name: entry_pointer,
#                 compare_entry[0]: compare_entry[1]
#             })})
#
#         return experiments
#
#     return create_gurobi_permutations(('random_relative', partial(RandomAlphasSGD, update_strategy_ptr=Relative_Step)))
#
#     # experiments = {
#     #     # 'weighted_tanh': OrderedDict({
#     #     #     'weighted_tanh_relative': partial(WeightedAlphasSGD, update_strategy_ptr=Relative_Step, activation=np.tanh),
#     #     #     'weighted_relative': partial(WeightedAlphasSGD, update_strategy_ptr=Relative_Step),
#     #     # }),
#     #     # 'random_tanh': OrderedDict({
#     #     #     'weighted_tanh_relative': partial(WeightedAlphasSGD, update_strategy_ptr=Relative_Step, activation=np.tanh),
#     #     #     'random_relative': partial(RandomAlphasSGD, update_strategy_ptr=Relative_Step),
#     #     # }),
#     #     'random_gurobi_relative': OrderedDict({
#     #         'gurobi_relative': partial(AlphasGurobiBased, update_strategy_ptr=Relative_Step),
#     #         'random_relative': partial(RandomAlphasSGD, update_strategy_ptr=Relative_Step),
#     #     }),
#     #
#     #     'random_sigmoid_relative': OrderedDict({
#     #         'random_relative': partial(RandomAlphasSGD, update_strategy_ptr=Relative_Step),
#     #         'weighted_sigmoid_relative': partial(WeightedAlphasSGD, update_strategy_ptr=Relative_Step,
#     #                                              activation=sigmoid),
#     #     }),
#     #     'random_sigmoid_absolute': OrderedDict({
#     #         'random_absolute': partial(RandomAlphasSGD, update_strategy_ptr=Absolute_Step),
#     #         'weighted_sigmoid_absolute': partial(WeightedAlphasSGD, update_strategy_ptr=Absolute_Step,
#     #                                              activation=sigmoid),
#     #     }),
#     #     # 'all_random_relative': OrderedDict({
#     #     #     'all_relative': partial(AllAlphasSGD, update_strategy_ptr=Relative_Step),
#     #     #     'random_relative': partial(RandomAlphasSGD, update_strategy_ptr=Relative_Step),
#     #     # }),
#     #     # 'all_tanh_relative': OrderedDict({
#     #     #     'all_relative': partial(AllAlphasSGD, update_strategy_ptr=Relative_Step),
#     #     #     'weighted_tanh_relative': partial(WeightedAlphasSGD, update_strategy_ptr=Relative_Step, activation=np.tanh),
#     #     # }),
#     #     'all_sigmoid_relative': OrderedDict({
#     #         'all_relative': partial(AllAlphasSGD, update_strategy_ptr=Relative_Step),
#     #         'sigmoid_relative': partial(WeightedAlphasSGD, update_strategy_ptr=Relative_Step,
#     #                                     activation=sigmoid),
#     #     }),
#     #     'sigmoid_absolute_relative': OrderedDict({
#     #         'sigmoid_relative': partial(AllAlphasSGD, update_strategy_ptr=Relative_Step),
#     #         'sigmoid_absolute': partial(WeightedAlphasSGD, update_strategy_ptr=Absolute_Step,
#     #                                     activation=sigmoid),
#     #     }),
#     #     'all_absolute_relative': OrderedDict({
#     #         'all_absolute': partial(AllAlphasSGD, update_strategy_ptr=Absolute_Step),
#     #         'all_absolute': partial(AllAlphasSGD, update_strategy_ptr=Relative_Step),
#     #     }),
#     # }
#     # return experiments


# def get_algorithms_list():
#     return [
#         {'random_relative': partial(RandomAlphasSGD, update_strategy_ptr=Relative_Step)},
#         {'gurobi_relative_20_1_1_0':
#              partial(AlphasGurobiBased, update_strategy_ptr=Relative_Step, random_threshold=20, use_relu=True,
#                      add_alpha_constraint=True, use_counter_example=False)},
#         {'gurobi_relative_20_1_1_1':
#              partial(AlphasGurobiBased, update_strategy_ptr=Relative_Step, random_threshold=20, use_relu=True,
#                      add_alpha_constraint=True, use_counter_example=True)},
#         {'gurobi_relative_20_0_1_1':
#              partial(AlphasGurobiBased, update_strategy_ptr=Relative_Step, random_threshold=20, use_relu=False,
#                      add_alpha_constraint=True, use_counter_example=True)},
#         {'gurobi_relative_20_0_0_0':
#              partial(AlphasGurobiBased, update_strategy_ptr=Relative_Step, random_threshold=20, use_relu=False,
#                      add_alpha_constraint=False, use_counter_example=False)}
#     ]


def get_all_algorithms():
    algorithms_ptrs = OrderedDict({
        'gurobi_base': partial(GurobiMultiLayer, use_relu=True, add_alpha_constraint=True, use_counter_example=True),
        'gurobi_random': partial(GurobiMultiLayerRandom, use_relu=True, add_alpha_constraint=True,
                                 use_counter_example=True, max_steps=3),
        'gurobi_IIS': partial(GurobiMultiLayerIIS, use_relu=True, add_alpha_constraint=True,
                              use_counter_example=True, max_steps=3)
    })

    return algorithms_ptrs


def get_model_path(path: str) -> str:
    if not os.path.exists(path):
        path = os.path.join(MODELS_FOLDER, path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)


def create_sbatch_files(folder_to_write):
    exps = get_all_algorithms()
    models = ['model_20classes_rnn4_rnn4_rnn4_fc32_fc32_fc32_0200.pkl',
              'model_20classes_rnn4_rnn4_rnn4_rnn4_fc32_fc32_fc32_0200.pkl',
              'model_20classes_rnn8_rnn8_fc32_fc32_0200.pkl',
              'model_20classes_rnn12_rnn12_fc32_fc32_fc32_fc32_0200.pkl',
              'model_20classes_rnn16_fc32_fc32_fc32_fc32_0100.pkl',
              'model_20classes_rnn8_rnn4_rnn4_fc32_fc32_fc32_fc32_0150.pkl']
    for exp in exps.keys():
        for model in models:
            model = get_model_path()
            exp_time = str(datetime.now()).replace(" ", "-")
            with open(os.path.join(folder_to_write, "run_" + exp + model + ".sh"), "w") as slurm_file:
                # job_output_rel_path = "slurm_{exp}_{exp_time}.out"
                job_output_rel_path = "slurm_{}_{}.out".format(exp, exp_time)
                slurm_file.write('#!/bin/bash\n')
                slurm_file.write('#SBATCH --job-name={}_{}_{}\n'.format(model, exp, exp_time))
                # slurm_file.write(f'#SBATCH --job-name={model}_{exp}_{exp_time}\n')
                slurm_file.write('#SBATCH --cpus-per-task=3\n')
                # slurm_file.write(f'#SBATCH --output={model}_{job_output_rel_path}\n')
                slurm_file.write('#SBATCH --output={}_{}\n'.format(model, job_output_rel_path))
                # slurm_file.write(f'#SBATCH --partition={partition}\n')
                slurm_file.write('#SBATCH --time=30:00:00\n')
                slurm_file.write('#SBATCH --mem-per-cpu=300\n')
                slurm_file.write('#SBATCH --mail-type=BEGIN,END,FAIL\n')
                slurm_file.write('#SBATCH --mail-user=yuvalja@cs.huji.ac.il\n')
                slurm_file.write('export LD_LIBRARY_PATH=/cd/usr/yuvalja/projects/Marabou\n')
                slurm_file.write('export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou\n')
                # slurm_file.write(f'python3 rnn_experiment/self_compare/experiment.py {exp} {model}\n')
                slurm_file.write('python3 rnn_experiment/self_compare/experiment.py {} {}\n'.format(exp, model))


def parse_input_strings():
    network_path = "model_20classes_rnn4_fc32_epochs40.h5"

    if len(sys.argv) > 1:
        print(sys.argv[1])
        if sys.argv[1] == 'create_sbatch':
            sbatch_folder = os.path.join(SBATCH_FOLDER)
            if not os.path.exists(sbatch_folder):
                os.mkdir(sbatch_folder)
            create_sbatch_files(sbatch_folder)
            print("created experiments in: {}".format(SBATCH_FOLDER))
            exit(0)
        elif sys.argv[1] == 'single':
            # If single, add another argument with the index in the get_algorithms_list function
            # algorithms_ptrs = get_algorithms_list()[int(sys.argv[2])]
            raise NotImplementedError
        elif sys.argv[1] == 'net':
            # If net, use the predefined all_algorithms
            network_path = sys.argv[2]
            algorithms_ptrs = get_all_algorithms()
        else:
            raise NotImplementedError
            algorithms_ptrs = get_algorithms()[sys.argv[1]]
            if algorithms_ptrs is None:
                exit(1)
        if len(sys.argv) > 2 and not str.isnumeric(sys.argv[2]):
            network_path = sys.argv[2]
    else:
        algorithms_ptrs = get_all_algorithms()
    return algorithms_ptrs, network_path


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)
    np.random.seed(9)

    algorithms_ptrs = get_all_algorithms()
    other_idx_method = [lambda x: np.argsort(x)[-i] for i in range(2, 7)]

    if len(sys.argv) > 1:
        if sys.argv[1] == 'controlled':
            if len(sys.argv) < 3:
                print('usuage: rnn_experiment/self_compare/experiment.py controlled [MODEL_PATH] [POINTS_PATH] [START_IDX OPTINAL]')
            model_path = sys.argv[2]
            points_path = sys.argv[3]
            start_idx = 0
            if len(sys.argv) > 4:
                start_idx = int(sys.argv[4])
            points = pickle.load(open(points_path, "rb"))
            n_iterations = 12
            radius = 0.01
            results = run_controlled_experiment(model_path, algorithms_ptrs, points, other_idx_method, radius, n_iterations, start_idx)
            exit(0)


    for model_path in ["models/model_20classes_rnn4_rnn4_fc32_fc32_fc32_fc32_epochs50.h5",
                       "models/model_20classes_rnn4_rnn4_fc32_fc32_fc32_fc32_fc32_epochs50.h5"
                       ]:
        points = pickle.load(open("pickles/points.pkl", "rb"))
        n_iterations = 4
        radius = 0
        results = run_controlled_experiment(model_path, algorithms_ptrs, points, other_idx_method, radius, n_iterations, 0,
                                            steps_num=5000)
        for k in results.keys():
            if 'gurobi' not in results[k]:
                continue
            exp_res = results[k]['gurobi']['result']
            if exp_res:
                print('SUCCESS')
                print(results[k])
                exit(1)
    exit(0)

    algorithms_ptrs, network_path = parse_input_strings()
    network_path = 'simple_model.h5'
    # np.random.seed(10)
    algorithms_ptrs = {
        'gurobi_base': partial(GurobiMultiLayer, use_relu=True, add_alpha_constraint=True, use_counter_example=True),
        'gurobi_random': partial(GurobiMultiLayerRandom, use_relu=True, add_alpha_constraint=True,
                                 use_counter_example=True, max_steps=15),
        'gurobi_IIS': partial(GurobiMultiLayerIIS, use_relu=True, add_alpha_constraint=True,
                              use_counter_example=True, max_steps=15)
    }

    network_path = "model_20classes_rnn2_fc32_epochs200.h5"

    df = run_random_experiment(network_path, algorithms_ptrs, mean=-2, var=18, n_iterations=25, radius=0.1,
                               steps_num=5000, num_points=1)
    # for t in range(5,15):
    #     counter = 0
    #     for _ in range(5):
    #         try:
    #             df = run_random_experiment(network_path, algorithms_ptrs,mean=1, var=1, n_iterations=12, radius=0.1,
    #                                        steps_num=2, num_points=1)
    #             counter += 1
    #         except ValueError as e:
    #             pass
    #     print("for t={} got {} fesiable  solutions".format(t, counter))
