from abc import ABC, abstractmethod

import numpy as np


class Discretization(ABC):
    @abstractmethod
    def get_x_space(self, x_min, x_max):
        pass

    @abstractmethod
    def get_y_space(self, y_min, y_max):
        pass


class StepsDiscretization(Discretization):
    def __init__(self, x_steps, y_steps):
        self.x_steps = x_steps
        self.y_steps = y_steps

    def get_x_space(self, x_min, x_max):
        return np.linspace(x_min, x_max, self.x_steps, endpoint=True)

    def get_y_space(self, y_min, y_max):
        return np.linspace(y_min, y_max, self.x_steps, endpoint=True)


class StepSizeDiscretization(Discretization):
    def __init__(self, x_step, y_step):
        self.x_step = x_step
        self.y_step = y_step

    def get_x_space(self, x_min, x_max):
        return np.arange(x_min, x_max, self.x_step)

    def get_y_space(self, y_min, y_max):
        return np.arange(y_min, y_max, self.x_step)
