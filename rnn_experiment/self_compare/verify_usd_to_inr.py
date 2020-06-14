import os
from functools import partial

from rnn_algorithms.GurobiBased import AlphasGurobiBased
from rnn_experiment.self_compare.experiment import run_one_comparison, MODELS_FOLDER
from maraboupy import MarabouCore
from maraboupy.keras_to_marabou_rnn import query, Predicate

# def run_one_comparison(in_tensor, radius, idx_max, other_idx, h5_file, n_iterations, algorithms_ptrs, steps_num=2500,
#                        return_dict=False):


CURRENCY_MODELS = os.path.join(MODELS_FOLDER, "usd_to_inr")

def get_h5_path():
    return os.path.join(CURRENCY_MODELS, "Vanilla_NonShift.h5")
# query(xlim: List[Tuple[float, float]], P: Optional[List[MarabouCore.Equation]],
# Q: List[MarabouCore.Equation], h5_file_path: str, algorithm_ptr, n_iterations = 10, steps_num = 5000):

def test_output_less_then_100():
    xlim = [(0, 10)]
    P = None
    # output[0] <= xlim[1] * 2
    less_then_twice = Predicate([(0, 1)], xlim[0][1] * 100, False, MarabouCore.Equation.LE)
    algo_ptr = partial(AlphasGurobiBased, use_relu=True, add_alpha_constraint=True, use_counter_example=True)
    res, queries_stats, alpha_history = query(xlim, P, [less_then_twice], get_h5_path(), algo_ptr)
    assert res


def test_output_less_then_80():
    xlim = [(0, 5)]
    P = None
    # output[0] <= xlim[1] * 2
    less_then_twice = Predicate([(0, 1)], xlim[0][1] * 80, False, MarabouCore.Equation.LE)
    algo_ptr = partial(AlphasGurobiBased, use_relu=True, add_alpha_constraint=True, use_counter_example=True)
    res, queries_stats, alpha_history = query(xlim, P, [less_then_twice], get_h5_path(), algo_ptr)
    assert res


def prove_output_less_then_twice():
    xlim = [(0, 3)]
    P = None
    # output[0] <= xlim[1] * 2
    fails = []
    passes = []
    for upper_bound in range(20,2,-1):
        for n in range(3,10):
            try:
                less_then_twice = Predicate([(0, 1)], xlim[0][1] * upper_bound, False, MarabouCore.Equation.LE)
                algo_ptr = partial(AlphasGurobiBased, use_relu=True, add_alpha_constraint=True, use_counter_example=True)
                res, queries_stats, alpha_history = query(xlim, P, [less_then_twice], get_h5_path(), algo_ptr, n_iterations=n)
                print("*" * 100)
                print("PASS: {},{}".format(upper_bound, n))
                print("*" * 100)
                passes.append((upper_bound, n))
                assert res
            except ValueError as e:
                fails.append((upper_bound, n))
                # print("*" * 100)
                # print("fail: {},{}".format(upper_bound, n))
                # print("*" * 100)
                # break

    print('PASS:')
    print("iterations, upper_bound:")
    print(["{}, {}".format(p[1], p[0] * xlim[0][1]) for p in passes])

    print('FAIL:')
    print("iterations, upper_bound:")
    print(["{}, {}".format(p[1], p[0] * xlim[0][1]) for p in fails])

if __name__ == "__main__":
    prove_output_less_then_twice()