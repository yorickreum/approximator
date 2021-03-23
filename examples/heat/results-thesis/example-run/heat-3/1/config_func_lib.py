import math

import torch

import approximator


def get_res(input: torch.Tensor, h_val: torch.Tensor):
    ones = torch.unsqueeze(torch.ones(len(input), dtype=approximator.DTYPE, device=approximator.DEVICE), 1)

    predicted_h_d = torch.autograd.grad(
        h_val,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    predicted_h_dt = predicted_h_d[:, 0:1]
    predicted_h_dz = predicted_h_d[:, 1:2]

    predicted_h_dz_d = torch.autograd.grad(
        predicted_h_dz,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]

    predicted_h_dz_dz = predicted_h_dz_d[:, 1:2]

    residual = predicted_h_dt - predicted_h_dz_dz
    return residual


def test_res():
    def h_analytically(input):
        t, z = input[:, 0:1], input[:, 1:2]
        return torch.exp(- math.pi ** 2 * t / 1600) * torch.sin(math.pi * z / 40)

    def h_wrong(input):
        t, z = input[:, 0:1], input[:, 1:2]
        return t * z

    input = torch.tensor([[42.12, 1.23]], requires_grad=True, dtype=approximator.DTYPE)
    prediction = h_wrong(input)
    res = get_res(input, prediction)
    print(res)


# test_res()
