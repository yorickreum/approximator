from inspect import signature
from typing import Callable

import torch


class Constraint:
    def __init__(self, condition: Callable, residual: callable, identifier=""):
        self.conditionf = condition
        self.__residualf = residual
        self.identifier = identifier

    def residualf(self, input: torch.tensor, prediction: torch.tensor):
        sig = signature(self.__residualf)
        if len(sig.parameters) == 2:
            return self.__residualf(input, prediction)
        elif len(sig.parameters) == 3:
            # with torch.no_grad():
            #     x, y = torch.tensor(input[:, 0], requires_grad=False), torch.tensor(input[:, 1], requires_grad=False)
            #     x, y = torch.unsqueeze(x, 1), torch.unsqueeze(y, 1)
            x, y = input[:, 0:1], input[:, 1:2]
            return self.__residualf(x, y, prediction)
        else:
            raise RuntimeError("Residual function not valid.")
