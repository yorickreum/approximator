import torch

import approximator

# alpha = 1.611e6  # cm
alpha = 0.019368480041686537  # m
theta_s = 0.287
theta_r = 0.075
beta = 3.96


def theta(h: torch.tensor):
    return ((alpha * (theta_s - theta_r)) / (alpha + torch.abs(h) ** beta)) + theta_r


Ks = 0.34  # in m per h
# A = 1.175e6 # cm
A = 0.0003890790677420439  # in m
gamma = 4.74


def K(h: torch.tensor):
    return Ks * A / (A + torch.abs(h) ** gamma)


def get_res(input: torch.Tensor, h_val: torch.Tensor):
    ones = torch.unsqueeze(torch.ones(len(input), dtype=approximator.DTYPE, device=approximator.DEVICE), 1)

    predicted_h_d = torch.autograd.grad(
        h_val,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    predicted_h_dz = predicted_h_d[:, 1:2]

    predicted_theta = theta(h_val)
    predicted_K = K(h_val)

    predicted_theta_d = torch.autograd.grad(
        predicted_theta,
        input,
        create_graph=True,
        grad_outputs=ones,
    )[0]
    predicted_theta_dt = predicted_theta_d[:, 0:1]

    predicted_second_term = predicted_K * predicted_h_dz
    predicted_second_term_d = torch.autograd.grad(
        predicted_second_term,
        input,
        create_graph=True,
        grad_outputs=ones,
    )[0]
    predicted_second_term_dz = predicted_second_term_d[:, 1:2]

    predicted_K_d = torch.autograd.grad(
        predicted_K,
        input,
        create_graph=True,
        grad_outputs=ones,
    )[0]
    predicted_K_dz = predicted_K_d[:, 1:2]

    # @TODO check the signs here
    residual = predicted_theta_dt - predicted_second_term_dz - predicted_K_dz
    return residual
