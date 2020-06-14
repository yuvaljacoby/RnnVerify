#!/usr/bin/env python

import math
import numpy as np
import sys
import argparse

# from keras.utils import plot_model

from gurobipy import *
from tensorflow.keras.models import load_model
from rns_verify.keras_rnn_checker import construct_rnn_from_keras_rnn

from timeit import default_timer as timer
from rns_verify.verify import VerificationHelper
from rns_verify.models import build_model
from rns_verify.rnn_abstractor import RNNAbstractor
from rns_verify.constants import *

# parser = argparse.ArgumentParser()
# parser.add_argument("-s", "--steps", default=4, type=int, help="The number of steps to use")
# parser.add_argument("-u", "--unroll", default="demand", choices=["start", "demand"], type=str, help="Unrolling method to use.")
# ARGS = parser.parse_args()

np.random.seed(1337)
UNROLL = 'demand'


def verify_query(model_path, in_tensor, y_idx_max, other_idx, n_iterations, radius):
    agent_model = load_model(model_path)
    keras_rnn = construct_rnn_from_keras_rnn(agent_model, built_myself=False)

    # Create the LP model and a wrapper to add constraints.
    gmodel = Model("Test")
    wrapper_gmodel = VerificationHelper(gmodel)
    gmodel.Params.LogToConsole = 0
    # gmodel.Params.DualReductions = 0
    gmodel.Params.MIPGap = 1e-6
    gmodel.Params.FeasibilityTol = 1e-7
    gmodel.Params.IntFeasTol = 1e-6

    start = timer()
    # Add the network inputs and the constraints on them.
    network_input = np.array([])
    for i in range(n_iterations):

        # TODO: We create here a bunch of new variables, and actually don't use previous variables again
        # I don't know why but when doing this + allowing the input to be in a radius getting UNSAT always
        # (also without the output constraints)
        if type(in_tensor) == np.ndarray:
            network_input = np.hstack((in_tensor, network_input))
        else:
            network_input = np.array(in_tensor * (i + 1))
        if UNROLL == 'start':
            abstractor = RNNAbstractor(keras_rnn, i + 1, abstraction_type=INPUT_ON_START_ONE_OUTPUT)
        else:
            abstractor = RNNAbstractor(keras_rnn, i + 1, abstraction_type=INPUT_ON_DEMAND_ONE_OUTPUT)
        agent_model_ffnn = abstractor.build_abstraction()

        dense, relu = wrapper_gmodel.add_vars(agent_model_ffnn.get_layers())
        gmodel.update()

        # Add the constraints for the network itself.

        layer_output = wrapper_gmodel.add_constraints(agent_model_ffnn.get_layers(), network_input, dense, relu, radius)

        # set constraints on the layer_output, for example here class 1 > class 0
        gmodel.addConstr((layer_output[y_idx_max] <= layer_output[other_idx]))

        # Add constraints for all the epsilons.
        epsilons = quicksum((quicksum(e) for (e, _, _) in dense))

        gmodel.addConstr(epsilons <= 1e-7)

    # Update the model.
    gmodel.update()
    # print('Finished adding constraints. Took {}s.'.format(end_constrs - start_constrs))

    gmodel.optimize()
    end = timer()
    # print("status infesiable?", gmodel.status == GRB.INFEASIBLE)
    # print("Time taken (verification): {}s".format(end - start))
    # print("Number of variables: {}".format(gmodel.NumVars))
    # print("Number of constraints: {}".format(gmodel.NumConstrs))
    assert gmodel.status == GRB.CUTOFF or gmodel.status == GRB.INFEASIBLE
    return end - start


def test_class5_1rnn2_0():
    model_path = 'models/model_classes5_1rnn2_0_64_4.h5'
    y_idx_max = 4
    other_idx = 0
    in_tensor = [10] * 40
    n_iterations = 10  # 1000?
    r = 0.01

    assert verify_query(model_path, in_tensor, y_idx_max, other_idx, n_iterations, r)
    assert not verify_query(model_path, in_tensor, other_idx, y_idx_max, n_iterations, r)


def test_class20_1rnn4_0():
    model_path = 'model_classes20_1rnn4_0_2_4.h5'
    y_idx_max = 13
    other_idx = 15
    in_tensor = [6.3, 9.4, 9.6, 3.1, 8.5, 9.4, 7.2, 8.6, 3.8, 1.4, 0.7, 7.8, 1.9, 8.2, 6.2, 3.6, 8.7, 1.7
        , 2.8, 4.8, 4.3, 5.1, 3.8, 0.8, 2.4, 7.6, 7.3, 0., 3.3, 7.4, 0., 2.1, 0.5, 8., 7.1, 3.9
        , 3., 8.3, 5.6, 1.8]
    n_iterations = 10
    r = 0

    assert not verify_query(model_path, in_tensor, other_idx, y_idx_max, n_iterations, r)
    assert verify_query(model_path, in_tensor, y_idx_max, other_idx, n_iterations, r)


def test_classes20_1rnn2_0():
    model_path = 'model_classes20_1rnn2_0_64_4.h5'
    y_idx_max = 9
    other_idx = 2
    in_tensor = [10] * 40
    n_iterations = 10
    r = 0.01

    assert not verify_query(model_path, in_tensor, other_idx, y_idx_max, n_iterations, r)
    assert verify_query(model_path, in_tensor, y_idx_max, other_idx, n_iterations, r)


if __name__ == "__main__":
    # TODO: Support dense layer after rnn
    test_class5_1rnn2_0()
    test_classes20_1rnn2_0()

    # Does not work, need to check why
    test_class20_1rnn4_0()
