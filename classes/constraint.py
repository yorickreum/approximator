from inspect import signature
from typing import Callable

import torch


class Constraint:
    def __init__(self, identifier: str, condition: Callable, residual: Callable, prepone=False):
        self.conditionf = condition
        self.__residualf = residual
        self.identifier = identifier
        self.prepone = prepone

    def __eq__(self, other):
        # return (self.__class__ == other.__class__
        #         and self.conditionf == other.conditionf
        #         and self.__residualf == other.__residualf
        #         and self.identifier == other.identifier
        #         and self.prepone == other.prepone)
        return (self.__class__ == other.__class__
                and self.identifier == other.identifier)

    def __hash__(self):
        return hash(self.identifier)

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

