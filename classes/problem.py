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
        constraint_identifiers = [c.identifier for c in constraints]
        if len(set(constraint_identifiers)) != len(constraint_identifiers):
            raise RuntimeError("Duplicated constraint identifier detected!")
        self._constraints = constraints

    def constraints(self, prepone_only=False):
        return [c for c in self._constraints if (prepone_only == False) or (c.prepone == True)]
