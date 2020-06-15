from typing import Tuple, List

from gurobipy import Model, LinExpr, GRB, Constr

from maraboupy import MarabouCore
from RNN.MarabouRnnModel import LARGE


class Bound:
    def __init__(self, gmodel: Model, upper: bool, initial_value: float, bound_idx: int, polyhedron_idx: int):
        self.upper = upper
        self.alpha = LARGE
        if not self.upper:
            self.alpha = -self.alpha
        self.initial_value = initial_value
        self.bound_idx = bound_idx
        self.polyhedron_idx = polyhedron_idx

        self.alpha_val = None
        self.beta_val = None

        # def attach_bound_to_model(self, gmodel: Model) -> None:
        self.gmodel = gmodel
        first_letter = 'u' if self.upper else 'l'
        self.name = "a{}{}^{}".format(first_letter, self.bound_idx, self.polyhedron_idx)
        self.alpha_var = gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name=self.name)
        self.beta_var = gmodel.addVar(lb=-LARGE, ub=LARGE, vtype=GRB.CONTINUOUS,
                                      name="b{}{}^{}".format(first_letter, self.bound_idx, self.polyhedron_idx))
        self.relu_bound = {} # Map between time and a relu constraint
        self.delta_relu_bound = {}

        init_constr_name = '{}_init_val'.format(self.name.replace('a', 'b'))
        if self.upper:
            gmodel.addConstr(self.beta_var >= initial_value, name=init_constr_name)
        else:
            gmodel.addConstr(self.beta_var <= initial_value, name=init_constr_name)

    def __eq__(self, other):
        if not isinstance(other, Bound):
            return False
        if self._was_model_optimized() and other._was_model_optimized():
            return self.alpha_val == other.alpha_val and self.beta_val == other.beta_val
        else:
            return self.name == other.name

    def get_relu(self, gmodel, t) -> LinExpr:
        if t in self.relu_bound:
            return self.relu_bound[t]
        else:
            first_letter = "u" if self.is_upper() else "l"
            cond = self.get_rhs(t)
            cond_f = gmodel.addVar(lb=0, ub=LARGE, vtype=GRB.CONTINUOUS, name=self.name + "_relu_output{}".format(t))
            delta = gmodel.addVar(vtype=GRB.BINARY)
            gmodel.addConstr(cond_f >= cond, name=self.name + "cond_relu0_t{}".format(t))
            gmodel.addConstr(cond_f <= cond + LARGE * delta, name=self.name + "cond_relu1_t{}".format(t))
            gmodel.addConstr(cond_f <= LARGE * (1 - delta), name=self.name + "cond_relu2_t{}".format(t))
            self.relu_bound[t] = cond_f
            self.delta_relu_bound[t] = delta

            return cond_f

    def get_rhs(self, t: int) -> LinExpr():
        if self.alpha_var is None:
            raise Exception("Should first attach to model")
        t = max(t, 0)
        return self.alpha_var * t + self.beta_var

    def get_lhs(self, t: int) -> LinExpr():
        if self.alpha_var is None:
            raise Exception("Should first attach to model")
        return self.alpha_var * t + self.beta_var

    def get_objective(self, alpha_weight=2, beta_weight=1) -> LinExpr():
        obj = self.alpha_var * alpha_weight + self.beta_var * beta_weight
        # we want the lower bound to be as tight as possible so we should prefer large numbers on small numbers
        if not self.is_upper():
            obj = -obj
        #     obj = self.alpha_var * alpha_weight * 3 + self.beta_var * beta_weight * -4
        # else:
        #     obj = self.alpha_var * (alpha_weight * -2) + self.beta_var * (beta_weight * -2)
        return obj

    def is_upper(self) -> bool:
        return self.upper

    def _was_model_optimized(self):
        return self.alpha_val is not None and self.beta_val is not None

    def __str__(self):
        if self._was_model_optimized():
            return '{}: {}*i + {}'.format(self.name, round(self.alpha_val, 6), round(self.beta_val, 6))
        else:
            return self.name

    def __repr__(self):
        return self.__str__()

    def get_equation(self, loop_idx, rnn_out_idx):
        if not self._was_model_optimized():
            raise Exception("First optimize the attached model")

        inv_type = MarabouCore.Equation.LE if self.is_upper() else MarabouCore.Equation.GE
        # if inv_type == MarabouCore.Equation.LE:
        #     ge_better = 1
        # else:
        #     ge_better = -1

        # is_upper:True  -> RNN_OUT <= alpha * i + beta <--> RNN_OUT - alpha * i <=  beta
        # is_upper:False -> RNN_OUT >= alpha * i + beta <--> RNN_OUT - alpha * i >=  beta
        invariant_equation = MarabouCore.Equation(inv_type)
        invariant_equation.addAddend(1, rnn_out_idx)  # b_i
        invariant_equation.addAddend(-self.alpha_val, loop_idx)  # i
        invariant_equation.setScalar(self.beta_val)
        return invariant_equation

    def model_optimized(self) -> 'Bound':
        self.alpha_val = self.alpha_var.x
        self.beta_val = self.beta_var.x

        return self

    def get_bound(self) -> Tuple[int, int]:
        return self.alpha_val, self.beta_val

    def get_iis_weight(self, gmodel: Model, iis_constrains: List[Constr]) -> int:
        weight_a, weight_b = 0, 0
        for c in iis_constrains:
            weight_a += abs(gmodel.getCoeff(c, self.alpha_var))
            weight_b += abs(gmodel.getCoeff(c, self.beta_var))

        return weight_a + weight_b
