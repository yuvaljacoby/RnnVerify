from gurobipy import *
import numpy as np


class VerificationHelper(object):
    def __init__(self, gmodel):
        self.gmodel = gmodel

    def _dense_vars(self, layer):
        output_size = layer.shape[1]
        epsilon_vars = []
        output_vars = []
        delta_vars = []
        for i in range(output_size):
            epsilon_vars.append(self.gmodel.addVar())
            output_vars.append(self.gmodel.addVar(lb=-GRB.INFINITY))
            delta_vars.append(self.gmodel.addVar(vtype=GRB.BINARY))
        return epsilon_vars, output_vars, delta_vars

    def _dense_vars_ffnn(self, layer):
        output_size = layer.output_shape[1]
        epsilon_vars = []
        output_vars = []
        delta_vars = []
        for i in range(output_size):
            epsilon_vars.append(self.gmodel.addVar())
            output_vars.append(self.gmodel.addVar(lb=-GRB.INFINITY))
            delta_vars.append(self.gmodel.addVar(vtype=GRB.BINARY))
        return epsilon_vars, output_vars, delta_vars

    def _dense_constraints(self, layer, epsilons, inputs_max, inputs_min, outputs):
        '''
        Add dense constraints, in the form:
            epsilon <= layer * inputs - outputs >= -epsilon
        :param outputs: gurobi variables
        '''
        output_size = layer.shape[1]
        weights = layer.T

        # Calculate the actual values
        dotted_outputs_min, dotted_outputs_max = [], []
        for i in range(output_size):
            dotted_outputs_max.append(np.array(weights[i]).dot(inputs_max))
            dotted_outputs_min.append(np.array(weights[i]).dot(inputs_min))


        # Add constraints on the outputs (which are variables), to make sure they are in the wanted range
        # epsilons are also variables, but in the optimization problem we make sure they are ~0
        for i in range(output_size):
            # output[i] >= dotted_outputs[i] - epsilon[i]
            self.gmodel.addConstr(dotted_outputs_min[i] - outputs[i] <= epsilons[i])
            # output[i] <= dotted_outputs[i] + epsilon[i]
            self.gmodel.addConstr(dotted_outputs_max[i] - outputs[i] >= -epsilons[i])

    def _dense_constraints_ffnn(self, layer, epsilons, inputs, outputs):
        output_size = layer.output_shape[1]
        weights = layer.get_weights()[0].T
        # bias = layer.get_weights()[1]
        dotted_outputs = [
            weights[i].dot(inputs) for i in range(output_size)
        ]
        for i in range(output_size):
            self.gmodel.addConstr(dotted_outputs[i] - outputs[i] <= epsilons[i])
            self.gmodel.addConstr(dotted_outputs[i] - outputs[i] >= -epsilons[i])

    def _relu_vars(self, relu_dimension):
        relu_vars = []
        for i in range(relu_dimension):
            relu_vars.append(self.gmodel.addVar())
        return relu_vars

    def _relu_vars_ffnn(self, layer):
        output_size = layer.output_shape[1]
        relu_vars = []
        for i in range(output_size):
            relu_vars.append(self.gmodel.addVar())
        return relu_vars

    def _relu_constraints(self, relu_dimension, pre, post, delta):
        '''
        For each relu (which is relu_dimension) add following constrains:
        post >= pre
        post <= pre + 1000 * delta
        post <= 1000 (1- delta)
        '''
        for i in range(relu_dimension):
            # self.gmodel.addGenConstrMax(post[i], [pre[i]], 0)
            self.gmodel.addConstr(post[i] >= pre[i])
            self.gmodel.addConstr(post[i] <= pre[i] + 1000 * delta[i])
            self.gmodel.addConstr(post[i] <= 1000 * (1 - delta[i]))

    def _relu_constraints_ffnn(self, layer, pre, post, delta):
        output_size = layer.output_shape[1]
        for i in range(output_size):
            # self.gmodel.addGenConstrMax(post[i], [pre[i]], 0)
            self.gmodel.addConstr(post[i] >= pre[i])
            self.gmodel.addConstr(post[i] <= pre[i] + 1000 * delta[i])
            self.gmodel.addConstr(post[i] <= 1000 * (1 - delta[i]))

    def add_vars(self, layers):
        dense, relu = [], []
        for i in range(len(layers)):
            dense.append(self._dense_vars(layers[i]))
            relu.append(self._relu_vars(layers[i].shape[1]))
        return dense, relu

    def add_vars_ffnn(self, layers):
        dense, relu = [], []
        for i in range(0, len(layers) - 1, 2):
            dense.append(self._dense_vars_ffnn(layers[i]))
            relu.append(self._relu_vars_ffnn(layers[i + 1]))
        dense.append(self._dense_vars_ffnn(layers[-1]))
        return dense, relu

    def add_constraints(self, layers, il, dense, relu, radius):

        for i in range(0, len(dense)):
            e, o, d = dense[i]
            r = relu[i]
            if i == 0:
                # Change the values only for the new inputs
                il_max = [v * (1 + radius) for v in il]
                il_min = [v * (1 - radius) for v in il]
            else:
                il_max = il
                il_min = il
            self._dense_constraints(layers[i], e, il_max, il_min, o)
            self._relu_constraints(layers[i].shape[1], o, r, d)
            il = r
        return o

    def add_constraints_ffnn(self, layers, il, dense, relu):
        for i in range(0, len(relu)):
            e, o, d = dense[i]
            r = relu[i]
            self._dense_constraints_ffnn(layers[2 * i], e, il, o)
            self._relu_constraints_ffnn(layers[2 * i + 1], o, r, d)
            il = r
        (e, o, _) = dense[-1]
        self._dense_constraints_ffnn(layers[-1], e, il, o)
        return o
