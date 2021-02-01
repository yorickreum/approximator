import time

import numpy as np
import torch
from torch import nn

import approximator
from approximator.classes.discretization import Discretization
from approximator.classes.problem import Problem


class Approximation:
    def __init__(self, problem: Problem, net: nn.Module):
        self.losses = []
        self.net = net
        if approximator.DTYPE == torch.double:
            self.net.double()
        self.net = self.net.to(device=approximator.DEVICE)
        self.problem = problem

    def train(self, learning_rate, epochs, discretization: Discretization, verbose_output=True):
        parameters = self.net.parameters()
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        if verbose_output:
            print("constraints")
            print([c.identifier for c in self.problem.constraints])

        start_epoches = time.time()
        try:
            for i in range(epochs):
                if verbose_output:
                    print("# step " + str(i) + ": #")
                optimizer.zero_grad()  # clear gradients for next train
                loss = self.loss_func(discretization=discretization, verbose_output=verbose_output)
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                self.losses += [loss.item()]
                # check if has converged, break early, @TODO
                # with torch.no_grad():
                #     if i % 10 == 0:
                #         if len(self.losses) >= 11:
                #             last_loss_changes = [
                #                 np.abs(a_i - b_i)
                #                 for a_i, b_i in zip(self.losses[-10:], self.losses[-11:-1])
                #             ]
                #             if all(llc <= torch.finfo(approximator.DTYPE).eps for llc in last_loss_changes):
                #                 # or: use max instead of all
                #                 break
        except KeyboardInterrupt:
            pass

        end_epoches = time.time()

        torch.save(self.net, f"./run/net.pt")
        # torch.save(self, "./run_old/approximation.pt")

        if len(self.losses) >= 2:
            print("loss difference: " + str(self.losses[-1] - self.losses[0]))
            print("loss decay: " + str((self.losses[-1] - self.losses[0]) / self.losses[0]))

    def loss_func(self, discretization: Discretization, verbose_output):
        constrained_inputs = discretization.get_spaces_for_constraints(self.problem)

        predictions = [self.net(constrained_input) for constrained_input in constrained_inputs]
        residuals = [self.problem.constraints[i].residualf(constrained_inputs[i], prediction)
                     for i, prediction in enumerate(predictions)]
        reduced_residuals = [torch.mean(residual) for residual in residuals]
        if verbose_output:
            print("mean residuals:")
            print("\n".join(
                [str(res.item()) + " for " + self.problem.constraints[i].identifier
                 for i, res in enumerate(reduced_residuals)])
            )
        loss = sum(reduced_residuals)
        if verbose_output:
            print("loss: " + str(loss.item()))
        return loss

    def load(self, path=None):
        if path is not None:
            self.net = torch.load(path, map_location=approximator.DEVICE)
        else:
            self.net = torch.load(f"./run/net.pt", map_location=approximator.DEVICE)

    def use(self, x: float, y: float):
        return self.net(torch.tensor([x, y], dtype=approximator.DTYPE, device=approximator.DEVICE)).item()

    def res(self, x: float, y: float):
        res = None
        for c in self.problem.constraints:
            if c.conditionf(x, y):
                input = torch.tensor([[x, y]], requires_grad=True, dtype=approximator.DTYPE, device=approximator.DEVICE)
                output = self.net(input)
                res = c.residualf(input, output).item()
        return res
