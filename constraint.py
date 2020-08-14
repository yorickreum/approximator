from typing import Callable

import torch


class Constraint:
    def __init__(self, condition: Callable, residual: callable):
        self.conditionf = condition
        self.__residualf = residual

    def residualf(self, input: torch.tensor, prediction: torch.tensor):
        with torch.no_grad():
            x, y = torch.tensor(input[:, 0], requires_grad=False), torch.tensor(input[:, 1], requires_grad=False)
            x, y = torch.unsqueeze(x, 1), torch.unsqueeze(y, 1)
        return self.__residualf(x, y, prediction)
