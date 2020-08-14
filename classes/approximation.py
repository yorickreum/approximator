import time

import torch

import approximator
from approximator.classes.discretization import Discretization
from approximator.classes.model import Model
from approximator.classes.problem import Problem


class Approximation:
    def __init__(self, problem: Problem, discretization: Discretization, n_hidden_layers, n_neurons_per_layer,
                 learning_rate,
                 epochs):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_neurons = n_neurons_per_layer
        self.n_hidden_layers = n_hidden_layers
        self.discretization = discretization
        self.problem = problem
        self.model = Model(problem=problem,
                           discretization=discretization,
                           n_hidden=self.n_hidden_layers,
                           n_neurons=self.n_neurons)

    def train(self):
        parameters = self.model.net.parameters()
        optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate)

        start_epoches = time.time()
        for i in range(self.epochs):
            optimizer.zero_grad()  # clear gradients for next train
            loss = self.model.loss_func()
            with torch.no_grad():
                if i % approximator.LOGSTEPS == 0:
                    print("step " + str(i) + ": ")
                    print(loss)
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            # check if has converged, break early
            with torch.no_grad():
                if i % 10 == 0:
                    if len(self.model.net.losses) >= 11:
                        last_loss_changes = [
                            torch.abs(a_i - b_i)
                            for a_i, b_i in zip(self.model.net.losses[-10:], self.model.net.losses[-11:-1])
                        ]
                        if all(llc <= torch.finfo(approximator.DTYPE).eps for llc in last_loss_changes):
                            # or: use max instead of all
                            break
        end_epoches = time.time()

        torch.save(self.model.net, "./run/model.pt")

    def use(self, x: float, y: float):
        return self.model.net(torch.tensor([x, y], dtype=approximator.DTYPE, device=approximator.DEVICE)).item()
