import random
from datetime import datetime
from itertools import product
from timeit import default_timer as timer
from typing import List, Tuple, Union, Dict

from gurobipy import LinExpr, Var, Model, Env, GRB, setParam

from RNN.MarabouRNNMultiDim import alpha_to_equation
from maraboupy import MarabouCore
from polyhedron_algorithms.GurobiBased.Bound import Bound

random.seed(0)
sign = lambda x: 1 if x >= 0 else -1

SMALL = 10 ** -4
LARGE = 10 ** 5
PRINT_GUROBI = False

setParam('Threads', 1)
setParam('NodefileStart', 0.5)
STAT_TOTAL_COUNTER = 'total_gurobi_steps'
STAT_FAIL_COUNTER = 'fail_gurobi_steps'
STAT_TOTAL_TIME = 'time_gurobi_steps'
STAT_IMPROVE_DECISION_TIME = 'time_gurobi_imporve_decision'
STAT_CONTINUOUS_COUNTER_EXAMPLE = 'continuous_counter_examples'



class GurobiSingleLayer:
    def __init__(self, rnnModel, xlim, polyhedron_max_dim, use_relu=True, use_counter_example=False,
                 add_alpha_constraint=False, max_steps=5,
                 layer_idx=0, debug=False, stats={}, **kwargs):
        '''

        :param rnnModel:
        :param xlim:
        '''
        self.w_in, self.w_h, self.b = rnnModel.get_weights()[layer_idx]
        self.rnnModel = rnnModel
        self.dim = self.w_h.shape[0]
        self.add_alpha_constraint = add_alpha_constraint
        self.use_counter_example = use_counter_example
        self.use_relu = use_relu
        self.stats = self.initialize_stats(stats)
        self.return_vars = True
        self.xlim = xlim
        self.initial_values = None
        self.n_iterations = rnnModel.n_iterations
        self.layer_idx = layer_idx
        self.prev_layer_beta = [None] * self.dim
        self.alphas_u = []
        self.alphas_l = []
        self.added_constraints = None
        self.polyhedron_max_dim = polyhedron_max_dim
        self.polyhedron_current_dim = 1
        assert self.polyhedron_max_dim >= self.polyhedron_current_dim
        self.step_num = 1
        self.same_step_counter = 0
        self.max_steps = max_steps
        rnn_start_idxs, rnn_output_idxs = rnnModel.get_start_end_idxs(layer_idx)
        self.rnn_output_idxs = rnn_output_idxs
        self.rnn_start_idxs = rnn_start_idxs + rnn_start_idxs
        self.rnn_output_idxs_double = rnn_output_idxs + rnn_output_idxs
        self.UNSAT = False
        self.debug = debug
        self.inv_type = [MarabouCore.Equation.GE] * self.dim + [MarabouCore.Equation.LE] * self.dim
        self.equations = [None] * self.dim * 2
        # self.actual_vars = []
        if xlim is not None:
            self.update_xlim([x[0] for x in xlim], [x[1] for x in xlim])

        self.t_options = set(range(self.n_iterations + 1))
        self.approximate_layers = False
        self.alphas = None
        # [Bound(False, self.initial_values[0][i], i) for i in range(self.dim)] + [Bound(True, self.initial_values[1][i], i) for i in range(self.dim)]
        # [-LARGE] * self.dim + [LARGE] * self.dim

        # assert len(self.alphas) == (2 * self.dim)
        # assert (2 * self.dim) == len(self.inv_type)
        assert (2 * self.dim) == len(self.rnn_output_idxs_double)

        self.last_fail = None
        self.alpha_history = []

    @staticmethod
    def initialize_stats(stats: Dict) -> Dict:
        if stats == {}:
            stats[STAT_TOTAL_COUNTER] = 0
            stats[STAT_FAIL_COUNTER] = 0
            stats[STAT_TOTAL_TIME] = 0
            stats[STAT_IMPROVE_DECISION_TIME] = 0
            stats[STAT_CONTINUOUS_COUNTER_EXAMPLE] = []
        return stats

    def __del__(self):
        return

    def update_xlim(self, lower_bound, upper_bound, beta=None):
        '''
        Update of the xlim, if it is from an invariant then lower_bound and upper_bound are time dependent and beta is not
        (i.e. lower_bound * t + beta <= V <= upper_bound*t + beta where V is the neuron value
        :param lower_bound: length self.dim of lower bounds
        :param upper_bound: length self.dim of upper bounds
        :param beta: length self.dim of scalars
        :return:
        '''
        assert len(lower_bound) == len(upper_bound)

        if isinstance(lower_bound[0], list):
            # TODO: This is good only if using the box (or polyhedron with n=1)
            assert False
            lower_bound = [l[0] for l in lower_bound]
            upper_bound = [u[0] for u in upper_bound]

        # The the alpha bounds for later use
        xlim = []
        for l, u in zip(lower_bound, upper_bound):
            xlim.append((l, u))
        self.xlim = xlim

        if beta is not None:
            assert len(beta[0]) == len(lower_bound)
            self.prev_layer_beta = beta
            # self.initial_values = self.calc_prev_layer_in_val(,t=0)

            self.initial_values = ([], [])
            assert (len(beta) == 2)
            assert (isinstance(beta[0], list))
            assert len(lower_bound) == len(upper_bound)
            assert len(lower_bound) == len(beta[0])
            assert len(lower_bound) == len(beta[1])

            # for i in range(len(lower_bound)):
            for i in range(self.dim):
                init_l, init_u = self.calc_prev_layer_in_val(i, 0)
                # self.initial_values[0].append(init_l + self.b[i])
                # self.initial_values[1].append(init_u + self.b[i])
                self.initial_values[0].append(init_l + self.b[i])
                self.initial_values[1].append(init_u + self.b[i])
            # initial_values = self.rnnModel.get_rnn_min_max_value_one_iteration(xlim, layer_idx=self.layer_idx,
            #                                                                    prev_layer_beta=beta)
            # self.initial_values = (self.initial_values[0], initial_values[1])

        else:
            initial_values = self.rnnModel.get_rnn_min_max_value_one_iteration(xlim, layer_idx=self.layer_idx,
                                                                               prev_layer_beta=beta)
            initial_values = (initial_values[0], initial_values[1])
            # initial_values = ([0] * len(initial_values[0]), initial_values[1])

            self.initial_values = initial_values

    def calc_prev_layer_in_val_inner_layer(self, i: int, t: int) -> (int, int):
        cond_x_u = 0
        cond_x_l = 0
        for j in range(self.w_in.shape[0]):
            assert self.xlim[j][1] * t + self.prev_layer_beta[1][j] >= 0
            w = self.w_in[j, i]
            if self.approximate_layers:
                # The input from the previous layer is an ReLU function output, therefore we take max with 0
                if w > 0:
                    cond_x_u += max((self.xlim[j][1] * t + self.prev_layer_beta[1][j]), 0) * w
                    cond_x_l += max((self.xlim[j][0] * t + self.prev_layer_beta[0][j]), 0) * w
                else:
                    cond_x_u += max((self.xlim[j][0] * t + self.prev_layer_beta[0][j]), 0) * w
                    cond_x_l += max((self.xlim[j][1] * t + self.prev_layer_beta[1][j]), 0) * w
            else:

                # if w > 0:
                #     cond_x_u += max((self.xlim[j][1] * self.n_iterations + self.prev_layer_beta[1][j]), 0) * w
                # else:
                #     # TODO: I am not sure why the outer max is necessary, but it helps to pass tests ....
                #     cond_x_u += max(max((self.xlim[j][0] * self.n_iterations + self.prev_layer_beta[0][j]), 0) * w, 0)

                # We have 4 options of update, use min for lower bound and max to upper
                x_update1 = max((self.xlim[j][0] * 0 + self.prev_layer_beta[0][j]), 0) * w
                x_update2 = max((self.xlim[j][1] * 0 + self.prev_layer_beta[1][j]), 0) * w
                x_update3 = max((self.xlim[j][0] * self.n_iterations + self.prev_layer_beta[0][j]), 0) * w
                x_update4 = max((self.xlim[j][1] * self.n_iterations + self.prev_layer_beta[1][j]), 0) * w
                cond_x_l += min(x_update1, x_update2, x_update3, x_update4)
                cond_x_u += max(x_update1, x_update2, x_update3, x_update4)
            # u_max = (self.xlim[j][1] * (self.n_iterations) + self.prev_layer_beta[1][j]) * w
            # u_min = (self.xlim[j][0] * (self.n_iterations) + self.prev_layer_beta[0][j]) * w
            # cond_x_u += max(u_max, u_min)
            # cond_x_l += min(u_max, u_min)
            #
            # l_max = self.prev_layer_beta[1][j] * w
            # l_min = self.prev_layer_beta[0][j] * w
            # cond_x_u = max(l_max, l_min, 0)
            # cond_x_l = min(l_max, l_min)
            # if u_max < u_min:
            #     cond_x_u += u_max
            #     cond_x_l += max(self.prev_layer_beta[0][j] * self.w_in[j, i], 0)  # t = 0
            # else:
            #     cond_x_u += u_min
            #     cond_x_l += max(self.prev_layer_beta[1][j] * self.w_in[j, i], 0)  # t = 0
        return cond_x_l, cond_x_u

    def calc_prev_layer_in_val(self, i: int, t: int) -> (int, int):
        cond_x_u = 0
        cond_x_l = 0
        # for j in range(len(self.xlim)):
        if self.prev_layer_beta[0] is not None:
            return self.calc_prev_layer_in_val_inner_layer(i, t)
        for j in range(self.w_in.shape[0]):
            v1 = self.xlim[j][1] * self.w_in[j, i]
            v2 = self.xlim[j][0] * self.w_in[j, i]
            if v1 > v2:
                cond_x_u += v1
                cond_x_l += v2
            else:
                cond_x_u += v2
                cond_x_l += v1

        return cond_x_l, cond_x_u

    def get_gurobi_rhs(self, gmodel, i: int, t: int, alphas_l: List[Bound], alphas_u: List[Bound]) -> (
            LinExpr, LinExpr):
        '''
        The upper bound for guroib is: alpha_u[0] >= w_h * t * (alpha_u[i] + initial) (for all i) + x + b
        :param i: The index on which we want the rhs
        :param t: time stamp
        :param alphas_l: gurobi variable for lower bound for each recurrent node
        :param alphas_u: gurobi variable for upper bound for each recurrent node
        :return: (cond_l, cond_u) each of type LinExpr
        '''

        cond_u = LinExpr()
        cond_l = LinExpr()

        for j in range(self.w_h.shape[1]):
            if True:
                if self.w_h[i, j] > 0:
                    cond_l += alphas_l[j].get_relu(gmodel, t) * self.w_h[i, j]
                    cond_u += alphas_u[j].get_relu(gmodel, t) * self.w_h[i, j]
                else:
                    cond_u += alphas_l[j].get_relu(gmodel, t) * self.w_h[i, j]
                    cond_l += alphas_u[j].get_relu(gmodel, t) * self.w_h[i, j]
            else:
                if self.w_h[i, j] > 0:
                    l_rhs_f, _ = self.get_relu_constraint(gmodel, alphas_l[j].get_rhs(t), 'cond_l',
                                                          'i{}t{}'.format(i, t),
                                                          False)
                    u_rhs_f, _ = self.get_relu_constraint(gmodel, alphas_u[j].get_rhs(t), 'cond_u',
                                                          'i{}t{}'.format(i, t),
                                                          True)
                    cond_l += l_rhs_f * self.w_h[i, j]
                    cond_u += u_rhs_f * self.w_h[i, j]
                else:
                    l_rhs_f, _ = self.get_relu_constraint(gmodel, alphas_u[j].get_rhs(t), 'cond_l',
                                                          'i{}t{}'.format(i, t),
                                                          False)
                    u_rhs_f, _ = self.get_relu_constraint(gmodel, alphas_l[j].get_rhs(t), 'cond_u',
                                                          'i{}t{}'.format(i, t),
                                                          True)
                    cond_l += l_rhs_f * self.w_h[i, j]
                    cond_u += u_rhs_f * self.w_h[i, j]

        cond_x_l, cond_x_u = self.calc_prev_layer_in_val(i, t)
        cond_u += cond_x_u + self.b[i]
        cond_l += cond_x_l + self.b[i]

        return cond_l, cond_u

    @staticmethod
    def get_relu_constraint(gmodel, cond: LinExpr, i: Union[int, str], t: Union[int, str], upper_bound: bool):
        first_letter = "u" if upper_bound else "l"
        cond_f = gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name="{}b{}_t_{}".format(first_letter, i, t))
        delta = gmodel.addVar(vtype=GRB.BINARY)
        gmodel.addConstr(cond_f >= cond, name="cond_{}_relu0_i{}_t{}".format(first_letter, i, t))
        gmodel.addConstr(cond_f <= cond + LARGE * delta, name="cond_{}_relu1_i{}_t{}".format(first_letter, i, t))
        gmodel.addConstr(cond_f <= LARGE * (1 - delta), name="cond_{}_relu2_i{}_t{}".format(first_letter, i, t))

        return cond_f, delta

    def add_disjunction_rhs(self, gmodel, conds: List[LinExpr], lhs: Var, greater: bool, cond_name: str):
        '''
        Add all the conditions in conds as a disjunction (using binary variables)
        :param gmodel: model to add condition to
        :param conds: list of rhs expressions
        :param lhs: lhs condition
        :param greater: if True them lhs >= rhs else lhs <= rhs
        :param cond_name: name to add in gurobi
        :return:
        '''
        deltas = []
        cond_delta = LinExpr()
        for cond in conds:
            deltas.append(gmodel.addVar(vtype=GRB.BINARY))
            cond_delta += deltas[-1]
            if greater:
                gmodel.addConstr(lhs - 2 * SMALL >= cond - (LARGE * deltas[-1]), name=cond_name)
            else:
                # lhs_f, _ = self.get_relu_constraint(gmodel, lhs, -1, -1, False)
                # gmodel.addConstr(lhs_f <= cond + (LARGE * deltas[-1]), name=cond_name)

                gmodel.addConstr(lhs + 2 * SMALL <= cond + (LARGE * deltas[-1]), name=cond_name)
            # gmodel.addConstr(deltas[-1] <= 0)

        gmodel.addConstr(cond_delta <= len(deltas) - 1, name="{}_deltas".format(cond_name))

    # def get_actual_values_vars(self, gmodel: Model, t: int) -> Tuple[List[Var], List[Var]]:
    #     def create_one_side(is_upper: bool) -> List[Var]:
    #         l = 'u' if is_upper else 'l'
    #         actual_vars_b = []  # vars before relu
    #         for i in range(self.w_h.shape[0]):
    #             b_var = gmodel.addVar(lb=-LARGE, ub=LARGE, vtype=GRB.CONTINUOUS, name='rb{}_{}^{}'.format(l, i, t))
    #             actual_vars_b.append(b_var)
    #             if is_upper:
    #                 for au in self.alphas_u[i]:
    #                     gmodel.addConstr(b_var >= au.get_rhs(t))
    #             else:
    #                 for al in self.alphas_l[i]:
    #                     gmodel.addConstr(b_var <= al.get_rhs(t))
    #
    #         actual_vars_f = []
    #         for i, b_var in enumerate(actual_vars_b):
    #             f_var = gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name='rf{}_{}^{}'.format(l, i, t))
    #             actual_vars_f.append(f_var)
    #             delta = gmodel.addVar(vtype=GRB.BINARY)
    #             gmodel.addConstr(f_var >= b_var, name="r{}_{}^{}_relu0".format(l, i, t))
    #             gmodel.addConstr(f_var <= b_var + LARGE * delta, name="r{}_{}^{}_relu1".format(l, i, t))
    #             gmodel.addConstr(f_var <= LARGE * (1 - delta), name="r{}_{}^{}_relu1".format(l, i, t))
    #         return actual_vars_f
    #
    #     actual_vars_f_l = create_one_side(False)
    #     actual_vars_f_u = create_one_side(True)
    #     return actual_vars_f_l, actual_vars_f_u

    def get_gurobi_polyhedron_model(self):
        env = Env()
        gmodel = Model("test", env)

        obj = LinExpr()
        self.alphas_l, self.alphas_u = self.set_gurobi_vars(gmodel)

        for hidden_idx in range(self.w_h.shape[0]):
            for a in self.alphas_l[hidden_idx]:
                obj += a.get_objective()
            for a in self.alphas_u[hidden_idx]:
                obj += a.get_objective()
        # print("*" * 100)
        # print("l: ", self.alphas_l)
        # print("u: ", self.alphas_u)
        # print("*"*100)
        gmodel.setObjective(obj, GRB.MINIMIZE)

        # To understand the next line use:
        #       [list(a_l) for a_l in product(*[[1,2],[3]])], [list(a_u) for a_u in product(*[[10],[20,30]])]
        all_alphas = [[list(a_l) for a_l in product(*self.alphas_l)], [list(a_u) for a_u in product(*self.alphas_u)]]

        for hidden_idx in range(self.w_h.shape[0]):
            for t in self.t_options:
                conds_l = []
                conds_u = []
                for alphas_l, alphas_u in zip(*all_alphas):
                    cond_l, cond_u = self.get_gurobi_rhs(gmodel, hidden_idx, t, alphas_l, alphas_u)
                    if self.use_relu:
                        cond_u, _ = self.get_relu_constraint(gmodel, cond_u, hidden_idx, t, upper_bound=True)
                        cond_l, _ = self.get_relu_constraint(gmodel, cond_l, hidden_idx, t, upper_bound=False)
                    conds_l.append(cond_l)
                    conds_u.append(cond_u)

                for _, al in enumerate(self.alphas_l[hidden_idx]):
                    self.add_disjunction_rhs(gmodel, conds=conds_l, lhs=al.get_lhs(t + 1),
                                             greater=False, cond_name="alpha_l{}_t{}".format(hidden_idx, t))
                for _, au in enumerate(self.alphas_u[hidden_idx]):
                    self.add_disjunction_rhs(gmodel, conds=conds_u, lhs=au.get_lhs(t + 1),
                                             greater=True, cond_name="alpha_u{}_t{}".format(hidden_idx, t))

                    # self.add_disjunction_rhs(gmodel, conds_l, alphas_l[hidden_idx] * (t + 1), False,
                    #                          "alpha_l{}_t{}".format(hidden_idx, t))
                    # self.add_disjunction_rhs(gmodel, conds_u, alphas_u[hidden_idx] * (t + 1), True,
                    #                          "alpha_u{}_t{}".format(hidden_idx, t))

        if not PRINT_GUROBI:
            gmodel.setParam('OutputFlag', False)

        if self.debug:
            gmodel.write("get_gurobi_polyhedron_model_step{}.lp".format(self.step_num))
        return env, gmodel

    def set_gurobi_vars(self, gmodel: Model) -> Tuple[List[List[Bound]], List[List[Bound]]]:
        alphas_u = []
        alphas_l = []
        for hidden_idx in range(self.w_h.shape[0]):
            alphas_l.append([])
            alphas_u.append([])
            for j in range(self.polyhedron_current_dim):
                cur_init_vals = (self.initial_values[0][hidden_idx], self.initial_values[1][hidden_idx])
                alphas_l[hidden_idx].append(Bound(gmodel, False, cur_init_vals[0], hidden_idx, j))
                alphas_u[hidden_idx].append(Bound(gmodel, True, cur_init_vals[1], hidden_idx, j))

        return alphas_l, alphas_u

    def improve_gurobi_model(self, gmodel: Union[Model, None]) -> bool:
        '''
        Got o model, extract anything needed to improve in the next iteration
        :param gmodel: infeasible model, None if the last mode was feasible but invariant failed
        :return wheater to do another step or not
        '''
        # gmodel.computeIIS()
        # gmodel.write("gurobi_improve.ilp")
        # if not self.approximate_layers:
        #     self.approximate_layers = True
        #     return True

        self.step_num += 1
        # self.polyhedron_current_dim += 1
        self.approximate_layers = False
        return self.step_num <= self.polyhedron_max_dim

    def gurobi_step_in_random_direction(self, previous_alphas, failed_improves=set()):
        valid_idx = [i for i in range(len(previous_alphas)) if i not in failed_improves and previous_alphas[i] != 0]
        if len(valid_idx) == 0:
            self.UNSAT = True
            return None, None, None
        else:
            idx = random.choice(valid_idx)
        assert previous_alphas[idx] != 0
        assert idx not in failed_improves

        # idx = np.random.randint(0, len(previous_alphas))
        # while previous_alphas[idx] == 0 or idx in failed_improves:
        #     idx = np.random.randint(0, len(previous_alphas))

        if idx < len(previous_alphas) / 2:
            return self.alphas_l[idx] >= previous_alphas[idx] * 2, "ce_output_alpha_l", idx
        else:
            print("adding constraint, alpha_u{} <= {}".format(idx - len(self.alphas_u), previous_alphas[idx] - SMALL))
            return self.alphas_u[idx - len(self.alphas_u)] <= previous_alphas[idx] - SMALL, 'ce_output_alpha_u', idx

    def do_gurobi_step(self, strengthen: bool, counter_examples=None, previous_alphas=None) -> \
            Union[List[int], List['Bound']]:
        self.stats[STAT_TOTAL_COUNTER] += 1
        start_time_step = timer()
        env, gmodel = self.get_gurobi_polyhedron_model()

        # Use counter example only when invariant failed (strengthen == False)
        if self.use_counter_example and not strengthen:
            if counter_examples:
                for i, (out, time) in enumerate(zip(*counter_examples)):
                    if out is None or time is None:
                        continue
                    if i < len(counter_examples[0]) // 2:
                        # It's a lower bound
                        for a in self.alphas_l[i]:
                            gmodel.addConstr(a.get_lhs(time + 1) <= out - (10 * SMALL))
                    else:
                        real_idx = i % (len(counter_examples[0]) // 2)
                        for a in self.alphas_u[real_idx]:
                                gmodel.addConstr(a.get_lhs(time - 1) >= out + (10 * SMALL))

        gmodel.optimize()

        status = gmodel.status
        error = None
        alphas = None
        if status == GRB.OPTIMAL:
            alphas = [[a.model_optimized() for a in ls] for ls in self.alphas_l] + [
                [a.model_optimized() for a in ls]
                for ls in self.alphas_u]
            print("{}: FEASIBLE alpahs = {}".format(str(datetime.now()).split(".")[0],
                                                    [str(a) for a_ls in alphas for a in a_ls]))
            # exit(1)

        elif status == GRB.INFEASIBLE or status == GRB.INF_OR_UNBD:
            error = ValueError("INFEASIBLE problem")
        else:
            # Not sure which other statuses can be ...
            assert False, status

        if error:
            start_time_improve = timer()
            improve = self.improve_gurobi_model(gmodel)
            end_time_improve = timer()
            self.stats[STAT_IMPROVE_DECISION_TIME] += end_time_improve - start_time_improve
            end_time_step = timer()
            self.stats[STAT_FAIL_COUNTER] += 1
            self.stats[STAT_TOTAL_TIME] += (end_time_step - start_time_step)
            print("FAIL Gurobi Step, {}, seconds: {}".format('retry' if improve else 'stop',
                                                             round(end_time_step - start_time_step, 3)))
            if improve:
                gmodel.dispose()
                env.dispose()
                return self.do_gurobi_step(True)
            elif self.added_constraints is not None:
                assert False
                # If the problem is infeasible and it's not the first try, add constraint and try again
                for con in self.added_constraints:
                    gmodel.remove(con)

                self.added_constraints = []
                # self.last_fail = None

                # TODO: Keep track on the recursion depth and use it for generating new bounds
                alphas = self.do_gurobi_step(strengthen, previous_alphas=self.alphas)
            else:
                # gmodel.computeIIS()
                # gmodel.write('get_gurobi_polyhedron_model_step1.ilp')
                self.UNSAT = True
                self.equations = None
                alphas = None

        # if self.UNSAT:
        #     self.equations = None
        #     return None

        # if alphas is None and not self.UNSAT:
        #     assert False
        end_time_step = timer()
        self.stats[STAT_TOTAL_TIME] += (end_time_step - start_time_step)
        gmodel.dispose()
        env.dispose()

        return alphas


    def update_all_equations(self):
        # initial_values = self.initial_values[0] + self.initial_values[1]
        self.equations = []
        if not isinstance(self.alphas[0], list):
            self.alphas = [[a] for a in self.alphas]

        for i, alpha in enumerate(self.alphas):
            self.equations.append([])
            for a in alpha:
                eq = a.get_equation(self.rnn_start_idxs[i], self.rnn_output_idxs_double[i])
                self.equations[-1].append(eq)


    def update_equation(self, idx):
        initial_values = self.initial_values[0] + self.initial_values[1]

        # Change the formal of initial values, first those for LE equations next to GE
        self.equations[idx] = alpha_to_equation(self.rnn_start_idxs[idx], self.rnn_output_idxs_double[idx],
                                                initial_values[idx], self.alphas[idx], self.inv_type[idx])


    def name(self):
        return 'gurobi_based_{}_{}'.format(self.alpha_initial_value, self.alpha_step_policy_ptr.__name__)


    def extract_equation_from_counter_example(self, counter_examples: List[Dict]):
        '''
        :param counter_examples: List of assingmnets marabou found as counter examples
        :return: outputs array, each cell is array of rnn_output values (as number of alpha_u), times the assingment
        for t  (len(times) == len(outputs)
        '''
        outputs = []
        times = []
        for i, counter_example in enumerate(counter_examples):
            if counter_example == {}:
                # We late user the index to update the correct invariant, so need to keep same length
                outputs.append(None)
                times.append(None)
                continue
            # We need to extract the time, and the values of all output indcies
            # Next create alpha_u >= time * output \land alpha_l \le time * output
            # outputs.append([counter_example[i] for i in self.rnn_output_idxs])
            outputs.append(counter_example[self.rnn_output_idxs_double[i]])
            # assert counter_example[self.rnn_start_idxs[0]] == counter_example[self.rnn_start_idxs[1]]
            times.append(counter_example[self.rnn_start_idxs[0]])
        return outputs, times


    def extract_hyptoesis_from_counter_example(self, counter_examples=[{}]):
        '''

        :param counter_examples: Array of assingmnets marabou found as counter examples
        :return: outputs array, each cell is array of memory cell values (as number of alpha_u), times the assingment
        for t  (len(times) == len(outputs)
        '''
        outputs = []
        times = []
        for counter_example in counter_examples:
            if counter_example == {}:
                continue
            # We need to extract the time, and the values of all output indcies
            # Next create alpha_u >= time * output \land alpha_l \le time * output
            outputs.append([counter_example[i - 1] for i in self.rnn_output_idxs])
            assert counter_example[self.rnn_start_idxs[0]] == counter_example[self.rnn_start_idxs[1]]
            times.append(counter_example[self.rnn_start_idxs[0]])
        return outputs, times


    def do_step(self, strengthen=True, invariants_results=[], sat_vars=None, layer_idx=0):
        '''
        do a step in the one of the alphas
        :param strengthen: determines the direction of the step if True will return a stronger suggestion to invert,
        weaker otherwise
        :return list of invariant equations (that still needs to be proven)
        '''

        if self.step_num > self.max_steps:
            assert False

        if invariants_results != [] and invariants_results is not None:
            pass
            # If we all invariants from above or bottom are done do step in the other
            # if all(min_invariants_results):
            #     self.next_is_max = True
            # elif all(max_invariants_results):
            #     self.next_is_max = False

        # TODO: If this condition is true it means the last step we did was not good, and we can decide what to do next
        #  (for example revert, and once passing all directions do a big step)
        if self.last_fail == strengthen:
            self.same_step_counter += 1
        else:
            self.same_step_counter = 0

        self.last_fail = strengthen

        # If the invariant failed we use two techinques to solve it: a. add the spesific counter example, b. add the time stemp
        counter_examples = None
        if not strengthen:
            print("{} Previous gurobi was not invariant, improve".format(str(datetime.now()).split(".")[0]))
            improve = False
            # First check if the invariant failed because of continues time
            if sat_vars:
                loop_idx = self.rnnModel.get_start_end_idxs()[0][0]
                for sat_var in sat_vars:
                    if round(sat_var.get(loop_idx, 0), 6) not in self.t_options:
                        improve = True
                        self.t_options.add(round(sat_var[loop_idx], 6))
                        self.stats[STAT_CONTINUOUS_COUNTER_EXAMPLE].append(sat_var[loop_idx])
                outputs, times = self.extract_equation_from_counter_example(sat_vars)
                counter_examples = (outputs, times)

        new_alphas = self.do_gurobi_step(strengthen, previous_alphas=self.alphas, counter_examples=counter_examples)
        if self.UNSAT:
            return None

        if new_alphas == self.alphas:
            # No improvement
            # TODO: What should we do in this case?
            # assert False
            return None

        if new_alphas is None:
            assert False
            # No fesabile solution, maybe to much over approximation, improve at random
            # TODO: Can we get out of this situation? fall back to something else or doomed to random?
        else:
            self.alphas = new_alphas

        self.update_all_equations()
        # print(self.alphas)
        self.alpha_history.append(self.get_alphas())

        return self.equations


    def revert_last_step(self):
        '''
        If last step did not work, call this to revert it (and then we still have a valid invariant)
        '''
        return


    def get_equations(self):
        if self.UNSAT:
            return None

        if self.equations[0] is None:
            # First call, first update the equations
            self.do_step(True)
        return self.equations


    def get_alphas(self):
        return self.alphas


    def get_bounds(self) -> Tuple[List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
        '''
        :return: (lower_bounds, upper_bounds), each bound is a list of (alpha,beta)
        '''
        if isinstance(self.alphas_l[0], list) and isinstance(self.alphas_l[0][0], Bound):
            return [[b.get_bound() for b in al] for al in self.alphas_l], \
                   [[b.get_bound() for b in au] for au in self.alphas_u]
        elif isinstance(self.alphas[0], list):
            # TODO: delete isinstance
            assert False
            alphas = [a[0] for a in self.alphas]
        else:
            # TODO: delete isinstance
            assert False
            alphas = self.alphas

        alphas_l = alphas[:len(alphas) // 2]
        alphas_u = alphas[len(alphas) // 2:]

        return ([[(a, b)] for a, b in zip(alphas_l, self.initial_values[0])],
                [[(a, b)] for a, b in zip(alphas_u, self.initial_values[1])])
        # else:
        #     raise Exception
