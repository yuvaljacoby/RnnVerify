from random import randint
from typing import Tuple, List

from gurobipy import Model

from polyhedron_algorithms.GurobiBased.Bound import Bound
from polyhedron_algorithms.GurobiBased.MultiLayerBase import GurobiMultiLayer
from polyhedron_algorithms.GurobiBased.SingleLayerBase import GurobiSingleLayer


class GurobiMultiLayerRandom(GurobiMultiLayer):
    def __init__(self, rnnModel, prev_layer_lim, max_steps=17, **kwargs):
        super(GurobiMultiLayerRandom, self).__init__(rnnModel, prev_layer_lim, max_steps = max_steps, **kwargs)
        # self.max_steps = max_steps

    def _initialize_single_layer(self, layer_idx: int) -> GurobiSingleLayer:
        if layer_idx == 0:
            prev_layer_lim = self.input_lim
        else:
            prev_layer_lim = None
        return GurobiSingleLayerRandom(self.rnnModel, prev_layer_lim, polyhedron_max_dim=self.polyhedron_max_dim,
                                       use_relu=self.use_relu, use_counter_example=self.use_counter_example,
                                       add_alpha_constraint=self.add_alpha_constraint, layer_idx=layer_idx,
                                       max_steps=self.max_steps, debug=self.debug)


class GurobiSingleLayerRandom(GurobiSingleLayer):
    def __init__(self, rnnModel, prev_layer_lim, **kwargs):
        super(GurobiSingleLayerRandom, self).__init__(rnnModel, prev_layer_lim, **kwargs)
        # GurobiSingleLayer.__init__(rnnModel, prev_layer_lim, **kwargs)
        self.max_steps = kwargs['max_steps']
        self.alphas_l_lengths = []
        self.alphas_u_lengths = []
        for hidden_idx in range(self.w_h.shape[0]):
            self.alphas_l_lengths.append(1)
            self.alphas_u_lengths.append(1)

    def set_gurobi_vars(self, gmodel: Model) -> Tuple[List[List['Bound']], List[List['Bound']]]:
        alphas_u = []
        alphas_l = []
        for hidden_idx in range(self.w_h.shape[0]):
            alphas_l.append([])
            alphas_u.append([])
            cur_init_vals = (self.initial_values[0][hidden_idx], self.initial_values[1][hidden_idx])
            for j in range(self.alphas_l_lengths[hidden_idx]):
                alphas_l[hidden_idx].append(Bound(gmodel, False, cur_init_vals[0], hidden_idx, j))
            for j in range(self.alphas_u_lengths[hidden_idx]):
                alphas_u[hidden_idx].append(Bound(gmodel, True, cur_init_vals[1], hidden_idx, j))

        return alphas_l, alphas_u

    def improve_gurobi_model(self, gmodel: Model) -> bool:
        '''
        Got o model, extract anything needed to improve in the next iteration
        :param gmodel: infeasible model
        :return wheater to do another step or not
        '''
        idx = randint(0, len(self.alphas_l) - 1)
        if self.step_num > 1:
            if randint(0, 1) == 0:
                self.alphas_l_lengths[idx] += 1
            else:
                self.alphas_u_lengths[idx] += 1

        print('step_num: {} out of: {}'.format(self.step_num, self.max_steps))
        print('lower bound lengths:', self.alphas_l_lengths)
        print('upper bound lengths:', self.alphas_u_lengths)
        self.step_num += 1
        # super(GurobiSingleLayerRandom, self).step_num += 1
        return self.step_num <= self.max_steps
