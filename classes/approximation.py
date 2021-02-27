import os
import time

import torch
from torch import nn

import approximator
from approximator.classes.discretization import Discretization
from approximator.classes.problem import Problem


class Approximation:
    def __init__(self, problem: Problem, net: nn.Module):
        self.losses = []
        self.latest_residuals = []
        self.net = net
        if approximator.DTYPE == torch.double:
            self.net.double()
        self.net = self.net.to(device=approximator.DEVICE)
        self.problem = problem
        self.pretraining_best_loss = None
        self.pretraining_best_loss_epoch = None
        self.pretraining_done = False
        self.training_best_loss = None
        self.training_best_loss_epoch = None

    def train(self, learning_rate, epochs, discretization: Discretization,
              target_loss=None,
              pretraining_target_loss=None,
              pretraining_patience=None,
              training_patience=None,
              checkpoint_dir_path=None,
              verbose_output=True):
        parameters = self.net.parameters()
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        if verbose_output:
            print("constraints")
            print([c.identifier for c in self.problem.constraints(False)])

        start_epoches = time.time()
        try:
            for i in range(epochs):
                if verbose_output:
                    print("# step " + str(i) + ": #")
                optimizer.zero_grad()  # clear gradients for next train

                pretraining_only = False
                if not self.pretraining_done:  # only check if pretraining not already done
                    # use target loss as criterion for pretraining
                    if pretraining_target_loss is not None:
                        # pre-training by loss
                        pretraining_only = not self.pretraining_done \
                                           and ((len(self.losses) == 0) or (self.losses[-1] > pretraining_target_loss))
                    # use patience as criterion for pretraining
                    elif (pretraining_patience is not None) and (checkpoint_dir_path is not None):
                        # pre-training by patience
                        if self.pretraining_best_loss_epoch is not None:
                            is_pretraining_patience_over = \
                                (i - self.pretraining_best_loss_epoch) > pretraining_patience
                        else:
                            is_pretraining_patience_over = False
                        pretraining_only = (self.pretraining_best_loss is None) or (not is_pretraining_patience_over)
                    # update flag
                    if not pretraining_only:
                        self.pretraining_done = True

                # print("last loss:")
                # print("none" if (len(self.losses) == 0) else str(self.losses[-1]) + " hence: " + str(
                #    (self.losses[-1] > prepone_loss)))
                loss = self.loss_func(discretization=discretization, verbose_output=verbose_output,
                                      prepone_only=pretraining_only)
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                self.losses += [loss.item()]
                if torch.isnan(loss):  # @TODO handle NaN
                    raise RuntimeError("NaN loss encountered!")

                if (not self.pretraining_done) and pretraining_only:
                    if (self.pretraining_best_loss is None) or (loss.item() < self.pretraining_best_loss):
                        self.pretraining_best_loss = loss.item()
                        self.pretraining_best_loss_epoch = i
                        self._save_model_checkpoint(checkpoint_dir_path)
                elif (self.training_best_loss is None) or (loss.item() < self.training_best_loss):
                    self.training_best_loss = loss.item()
                    self.training_best_loss_epoch = i
                    self._save_model_checkpoint(checkpoint_dir_path)

                # check if has converged, break early
                # via target loss
                if (target_loss is not None) and \
                        (pretraining_target_loss is None or self.pretraining_done) and \
                        (loss.item() < target_loss):
                    break
                # via patience
                elif (training_patience is not None) and (self.training_best_loss is not None):
                    is_training_patience_over = (i - self.training_best_loss_epoch) > training_patience
                    if is_training_patience_over:
                        self._load_model_checkpoint(checkpoint_dir_path)
                        print(f"Patience exceeded, best checkpoint model with loss {self.training_best_loss} loaded.")
                        break
        except KeyboardInterrupt:
            pass

        end_epoches = time.time()

        torch.save(self.net, f"./run/net.pt")
        # torch.save(self, "./run_old/approximation.pt")

        if len(self.losses) >= 2:
            print("loss difference: " + str(self.losses[-1] - self.losses[0]))
            print("loss decay: " + str((self.losses[-1] - self.losses[0]) / self.losses[0]))

    def _save_model_checkpoint(self, checkpoint_path):
        torch.save(self.net, os.path.join(checkpoint_path, f"checkpoint.pt"))

    def _load_model_checkpoint(self, checkpoint_path):
        self.load(os.path.join(checkpoint_path, f"checkpoint.pt"))

    def loss_func(self, discretization: Discretization, verbose_output, prepone_only):

        all_constraints = self.problem.constraints(prepone_only=False)

        active_constraints = self.problem.constraints(prepone_only)

        constrained_inputs = discretization.get_spaces_for_constraints(self.problem.domain, all_constraints)
        constrained_predictions = {constraint: self.net(constrained_inputs[constraint]) for constraint in
                                   all_constraints}

        all_residuals = {c: c.residualf(constrained_inputs[c], constrained_predictions[c]) for c in all_constraints}

        all_residuals_reduced = {c: torch.mean(all_residuals[c]) for c in all_constraints}

        self.latest_residuals = all_residuals_reduced.values()  # to extract

        if verbose_output:
            print("mean residuals:")
            print("\n".join(
                ["for " +
                 constraint.identifier +
                 " \t (gets considered? " + (
                     "[+]" if (constraint in active_constraints) else "[-]") + ")" +
                 ": \t" + str(all_residuals_reduced[constraint].item())
                 for constraint in all_constraints])
            )

        loss = sum([all_residuals_reduced[c] for c in active_constraints])

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
        for c in self.problem.constraints(False):
            if c.conditionf(x, y):
                input = torch.tensor([[x, y]], requires_grad=True, dtype=approximator.DTYPE, device=approximator.DEVICE)
                output = self.net(input)
                res = c.residualf(input, output).item()
        return res
