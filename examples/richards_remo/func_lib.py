import torch

import approximator

# constant variables
# fksat = saturated hydraulic conductivity [m/s]
# fmpot = saturated matric potential [m]
# bclapp = clapp-hornberger-parameter [-]
# vpor = pore volume [m/m]

# i, j = 60, 70  # 60, 70 --> lat = 53.275978, lon = 10.553134

# layer 1:
# b = 3.8787098  # bclapp = clapp-hornberger-parameter [-], ds_bclapp_1.variables["var9921"][:,i,j]
# theta_sat = 0.44415206  # vpor = pore volume [m/m], @TODO eventuell sollte hier ZWSAT / ZWSFC genutzt werden?
# h_sat = -0.07865309  # fmpot = saturated matric potential [m], ds_fmpot_1.variables["var9911"][:,i,j]
# K_sat_ms = 0.01837207  # fksat = saturated hydraulic conductivity [m/s], ds_fksat_1.variables["var9901"][:,i,j]
# K_sat = (7 * 24 * 60 * 60) * K_sat_ms  # [m/week]

# layer 2:
b = 3.9564855  # bclapp = clapp-hornberger-parameter [-]
theta_sat = 0.4165416  # vpor = pore volume [m/m]
h_sat = -0.07915318  # fmpot = saturated matric potential [m]
K_sat_ms = 0.013824694  # fksat = saturated hydraulic conductivity [m/s]
K_sat = (7 * 24 * 60 * 60) * K_sat_ms  # [m/week]


def theta_by_h(h: torch.tensor):  # Campbell (1974), h ist psi
    return theta_sat * (h / h_sat) ** (-1 / b)


def h_by_theta(theta: torch.tensor):  # Campbell (1974), h ist psi, inverse function
    return h_sat * (theta / theta_sat) ** (-b)


def K(theta: torch.tensor):  # Campbell (1974)
    return K_sat * (theta / theta_sat) ** (2 * b + 3)


def flux_Q(input: torch.Tensor, h_val: torch.Tensor):
    ones = torch.unsqueeze(torch.ones(len(input), dtype=approximator.DTYPE, device=approximator.DEVICE), 1)
    predicted_K = K(theta_by_h(h_val))
    predicted_h_d = torch.autograd.grad(
        h_val,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    predicted_h_dz = predicted_h_d[:, 1:2]
    return - predicted_K * predicted_h_dz - predicted_K


def get_res(input: torch.Tensor, h_val: torch.Tensor):
    ones = torch.unsqueeze(torch.ones(len(input), dtype=approximator.DTYPE, device=approximator.DEVICE), 1)

    # dTheta / dt
    predicted_theta = theta_by_h(h_val)
    predicted_theta_d = torch.autograd.grad(
        predicted_theta,
        input,
        create_graph=True,
        grad_outputs=ones,
    )[0]
    predicted_theta_dt = predicted_theta_d[:, 0:1]

    # d/dz ( K(theta) * dh/dz)
    predicted_h_d = torch.autograd.grad(
        h_val,
        input,
        create_graph=True,
        grad_outputs=ones
    )[0]
    predicted_h_dz = predicted_h_d[:, 1:2]

    predicted_K = K(predicted_theta)

    predicted_second_term = predicted_K * predicted_h_dz
    predicted_second_term_d = torch.autograd.grad(
        predicted_second_term,
        input,
        create_graph=True,
        grad_outputs=ones,
    )[0]
    predicted_second_term_dz = predicted_second_term_d[:, 1:2]

    # dK/dz
    predicted_K_d = torch.autograd.grad(
        predicted_K,
        input,
        create_graph=True,
        grad_outputs=ones,
    )[0]
    predicted_K_dz = predicted_K_d[:, 1:2]

    # @TODO check the signs here, because of z convention
    residual = predicted_theta_dt - predicted_second_term_dz - predicted_K_dz
    return residual
