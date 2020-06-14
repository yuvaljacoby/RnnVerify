from typing import Tuple, List

from gurobipy import Model, Var, Constr

from polyhedron_algorithms.GurobiBased.Bound import Bound
from polyhedron_algorithms.GurobiBased.MultiLayerBase import GurobiMultiLayer
from polyhedron_algorithms.GurobiBased.SingleLayerBase import GurobiSingleLayer


class GurobiMultiLayerIIS(GurobiMultiLayer):
    def __init__(self, rnnModel, prev_layer_lim, **kwargs):
        super(GurobiMultiLayerIIS, self).__init__(rnnModel, prev_layer_lim, **kwargs)

    def _initialize_single_layer(self, layer_idx: int) -> GurobiSingleLayer:
        if layer_idx == 0:
            prev_layer_lim = self.input_lim
        else:
            prev_layer_lim = None
        return GurobiSingleLayerIIS(self.rnnModel, prev_layer_lim, polyhedron_max_dim=self.polyhedron_max_dim,
                                       use_relu=self.use_relu, use_counter_example=self.use_counter_example,
                                       add_alpha_constraint=self.add_alpha_constraint, layer_idx=layer_idx,
                                       max_steps=self.max_steps)


class GurobiSingleLayerIIS(GurobiSingleLayer):
    def __init__(self, rnnModel, prev_layer_lim, **kwargs):
        super(GurobiSingleLayerIIS, self).__init__(rnnModel, prev_layer_lim, **kwargs)
        # GurobiSingleLayer.__init__(rnnModel, prev_layer_lim, **kwargs)
        self.max_steps = kwargs['max_steps']
        self.alphas_l_lengths = []
        self.alphas_u_lengths = []
        for hidden_idx in range(self.w_h.shape[0]):
            self.alphas_l_lengths.append(1)
            self.alphas_u_lengths.append(1)

    # def get_gurobi_polyhedron_model(self):
    #     from gurobipy import Env, LinExpr, GRB
    #     from itertools import product
    #
    #     env = Env()
    #     gmodel = Model("test", env)
    #
    #     obj = LinExpr()
    #     self.alphas_l, self.alphas_u = self.set_gurobi_vars(gmodel)
    #
    #     # for hidden_idx in range(self.w_h.shape[0]):
    #     #     for a in self.alphas_l[hidden_idx]:
    #     #         obj += a.get_objective()
    #     #     for a in self.alphas_u[hidden_idx]:
    #     #         obj += a.get_objective()
    #
    #
    #
    #     # To understand the next line use:
    #     #       [list(a_l) for a_l in product(*[[1,2],[3]])], [list(a_u) for a_u in product(*[[10],[20,30]])]
    #     all_alphas = [[list(a_l) for a_l in product(*self.alphas_l)], [list(a_u) for a_u in product(*self.alphas_u)]]
    #     for hidden_idx in range(self.w_h.shape[0]):
    #         for t in range(self.n_iterations):
    #             conds_l = []
    #             conds_u = []
    #             for alphas_l, alphas_u in zip(*all_alphas):
    #                 cond_l, cond_u = self.get_gurobi_rhs(hidden_idx, t, alphas_l, alphas_u)
    #                 if self.use_relu:
    #                     cond_u, _ = self.get_relu_constraint(gmodel, cond_u, hidden_idx, t, upper_bound=True)
    #                     cond_l, _ = self.get_relu_constraint(gmodel, cond_l, hidden_idx, t, upper_bound=False)
    #                 conds_l.append(cond_l)
    #                 conds_u.append(cond_u)
    #
    #
    #             for i, al in enumerate(self.alphas_l[hidden_idx]):
    #                 if i % 2:
    #                     obj += al.get_objective()
    #                 else:
    #                     obj -= al.get_objective()
    #                 self.add_disjunction_rhs(gmodel, conds=conds_l, lhs=al.get_lhs(t),
    #                                          greater=False, cond_name="alpha_l{}_t{}".format(i, t))
    #             for i, au in enumerate(self.alphas_u[hidden_idx]):
    #                 if i % 2:
    #                     obj += al.get_objective()
    #                 else:
    #                     obj -= al.get_objective()
    #                 self.add_disjunction_rhs(gmodel, conds=conds_u, lhs=au.get_lhs(t),
    #                                          greater=True, cond_name="alpha_u{}_t{}".format(i, t))
    #
    #     gmodel.setObjective(obj, GRB.MINIMIZE)
    #     return env, gmodel

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

        # print('step_num:', self.step_num)
        self.step_num += 1
        return alphas_l, alphas_u

    @staticmethod
    def _get_all_weights(gmodel: Model, iis_constraints: List[Constr], alphas: List[List[Var]]) -> List[int]:
        weights = []
        for a_ls in alphas:
            a_sum = 0
            for a in a_ls:
                a_sum += a.get_iis_weight(gmodel, iis_constraints)
            weights.append(a_sum)

        return weights

    def improve_gurobi_model(self, gmodel: Model) -> bool:
        '''
        Got o model, extract anything needed to improve in the next iteration
        :param gmodel: infeasible model
        :return wheater to do another step or not
        '''


        iis_constrains = []
        gmodel.write("gmodel_steps{}.lp".format(self.step_num))
        gmodel.computeIIS()
        gmodel.write("gmodel_iis_steps{}.ilp".format(self.step_num))
        self.alphas_l_lengths = [self.step_num] * len(self.alphas_l_lengths)
        self.alphas_u_lengths = [self.step_num] * len(self.alphas_u_lengths)
        if self.step_num > self.max_steps:
            return False

        return True

        for c in gmodel.getConstrs():
            if c.IISConstr:
                iis_constrains.append(c)

        lower_bound_weight = self._get_all_weights(gmodel, iis_constrains, self.alphas_l)
        upper_bound_weight = self._get_all_weights(gmodel, iis_constrains, self.alphas_u)
        assert len(lower_bound_weight) == len(upper_bound_weight)

        argmax = lambda l: max(range(len(l)), key=lambda i: l[i])
        self.alphas_l_lengths[argmax(lower_bound_weight)] += 1
        self.alphas_u_lengths[argmax(upper_bound_weight)] += 1
        # if self.step_num > 1:
        #     if randint(0, 1) == 0:
        #         self.alphas_l_lengths[idx] += 1
        #     else:
        #         self.alphas_u_lengths[idx] += 1

        return True
