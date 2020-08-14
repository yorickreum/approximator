from typing import List

from approximator.classes.constraint import Constraint


class Domain:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.y_max = y_max
        self.y_min = y_min
        self.x_max = x_max
        self.x_min = x_min


class Problem:
    def __init__(self, domain: Domain, constraints: List[Constraint]):
        self.domain = domain
        self.constraints = constraints
