import itertools
from abc import ABC, abstractmethod

import numpy as np
import torch

import approximator
from approximator.classes.problem import Domain


class Discretization(ABC):
    def __init__(self, x_additional=None, y_additional=None):
        if x_additional is None:
            x_additional = []
        if y_additional is None:
            y_additional = []
        self.x_additional = x_additional
        self.y_additional = y_additional

    @abstractmethod
    def get_x_space(self, x_min, x_max):
        pass

    @abstractmethod
    def get_y_space(self, y_min, y_max):
        pass

    def get_spaces_for_constraints(self, domain, constraints):
        pass


class AbstractStepsDiscretization(Discretization, ABC):
    def __init__(self, x_steps, y_steps, x_additional=None, y_additional=None):
        super().__init__(x_additional=x_additional, y_additional=y_additional)
        self.x_steps = x_steps
        self.y_steps = y_steps

    def calc_spaces_for_constraints(self, domain: Domain, constraints):
        constrained_spaces = {}
        # x_space, y_space = \
        #     self.get_x_space(problem.domain.x_min, problem.domain.x_max), \
        #     self.get_y_space(problem.domain.y_min, problem.domain.y_max)
        x_space, y_space = \
            self.get_x_space(domain.x_min, domain.x_max), \
            self.get_y_space(domain.y_min, domain.y_max)
        for constraint in constraints:  # problem.constraints(prepone_only):
            constrained_input = []
            for r in itertools.product(x_space, y_space):
                x, y = r[0], r[1]
                if constraint.conditionf(x, y):
                    constrained_input += [[x, y]]
            if len(constrained_input) == 0:
                raise RuntimeWarning("Constraint condition was never met in discretized domain!")
            constrained_spaces[constraint] = torch.tensor(
                constrained_input,
                dtype=approximator.DTYPE,
                requires_grad=True,
                device=approximator.DEVICE)

        if len(constrained_spaces) > 1:
            intersection = set.intersection(*[set(ci) for ci in constrained_spaces.values()])
            if len(intersection) > 0:
                raise RuntimeWarning("Multiple constraint conditions apply for at least one point, "
                                     "this can lead to unpredictable behaviour.")
        return constrained_spaces


class StepsDiscretization(AbstractStepsDiscretization):
    def __init__(self, x_steps, y_steps, x_additional=None, y_additional=None):
        super().__init__(x_steps, y_steps, x_additional, y_additional)
        self.x_space = None
        self.y_space = None
        # Cache calculated constrained inputs for speed
        self.cached_constraints = {}

    @staticmethod
    def get_lin_space(min, max, steps):
        return np.linspace(min, max, steps, endpoint=True)

    def get_x_space(self, x_min, x_max):
        if self.x_space is None:
            self.x_space = list({*StepsDiscretization.get_lin_space(x_min, x_max, self.x_steps), *self.x_additional})
        return self.x_space

    def get_y_space(self, y_min, y_max):
        if self.y_space is None:
            self.y_space = list({*StepsDiscretization.get_lin_space(y_min, y_max, self.y_steps), *self.y_additional})
        return self.y_space

    def get_spaces_for_constraints(self, rectangle, constraints):
        if list(self.cached_constraints.keys()) != constraints:
            self.cached_constraints = self.calc_spaces_for_constraints(rectangle, constraints)
        return self.cached_constraints


class RandomStepsDiscretization(AbstractStepsDiscretization):
    @staticmethod
    def get_random_space(min, max, steps):
        return [min + (max - min) * np.random.random() for _ in range(steps)]

    def get_x_space(self, x_min, x_max):
        return list({*RandomStepsDiscretization.get_random_space(x_min, x_max, self.x_steps), *self.x_additional})

    def get_y_space(self, y_min, y_max):
        return list({*RandomStepsDiscretization.get_random_space(y_min, y_max, self.y_steps), *self.y_additional})

    # def get_spaces_for_constraints(self, problem: Problem, prepone_only=False):
    def get_spaces_for_constraints(self, rectangle, constraints):
        return self.calc_spaces_for_constraints(rectangle, constraints)


class StepSizeDiscretization(Discretization):
    def __init__(self, x_step, y_step, x_additional=None, y_additional=None):
        super().__init__(x_additional=x_additional, y_additional=y_additional)
        self.x_step = x_step
        self.y_step = y_step
        # Cache calculated constrained inputs for speed
        self.cached_constraints = {}

    @staticmethod
    def get_arranged_space(min, max, step):
        return np.arange(min, max, step)

    def get_x_space(self, x_min, x_max):
        return list({*StepSizeDiscretization.get_arranged_space(x_min, x_max, self.x_step), *self.x_additional})

    def get_y_space(self, y_min, y_max):
        return list({*StepSizeDiscretization.get_arranged_space(y_min, y_max, self.y_step), *self.y_additional})

    def calc_spaces_for_constraints(self, domain: Domain, constraints):
        constrained_spaces = {}
        # x_space, y_space = \
        #     self.get_x_space(problem.domain.x_min, problem.domain.x_max), \
        #     self.get_y_space(problem.domain.y_min, problem.domain.y_max)
        x_space, y_space = \
            self.get_x_space(domain.x_min, domain.x_max), \
            self.get_y_space(domain.y_min, domain.y_max)
        for constraint in constraints:  # problem.constraints(prepone_only):
            constrained_input = []
            for r in itertools.product(x_space, y_space):
                x, y = r[0], r[1]
                if constraint.conditionf(x, y):
                    constrained_input += [[x, y]]
            if len(constrained_input) == 0:
                raise RuntimeWarning("Constraint condition was never met in discretized domain!")
            constrained_spaces[constraint] = torch.tensor(
                constrained_input,
                dtype=approximator.DTYPE,
                requires_grad=True,
                device=approximator.DEVICE)

        if len(constrained_spaces) > 1:
            intersection = set.intersection(*[set(ci) for ci in constrained_spaces.values()])
            if len(intersection) > 0:
                raise RuntimeWarning("Multiple constraint conditions apply for at least one point, "
                                     "this can lead to unpredictable behaviour.")
        return constrained_spaces

    def get_spaces_for_constraints(self, rectangle, constraints):
        if list(self.cached_constraints.keys()) != constraints:
            self.cached_constraints = self.calc_spaces_for_constraints(rectangle, constraints)
        return self.cached_constraints
