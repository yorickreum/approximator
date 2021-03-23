import os

import torch

import approximator
from approximator.classes.approximation import Approximation
from approximator.examples.richards_remo.func_lib import theta_by_h, res_layer_2_theta_t0, res_layer_2_theta_t1
from approximator.examples.richards_remo.richards import problem, approximation_net

approximation = Approximation(
    problem=problem,
    net=approximation_net
)


def load_trained_model(trained_model):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    approximation.load(os.path.join(dir_path, f"trained_models/{trained_model}/net.pt"))


def integrate_matric_potential(trained_model: int):
    raise NotImplementedError("It is probably useless to integrate? Mean should make more sense, so see below.")
    load_trained_model(trained_model)
    t_space = torch.linspace(0, 4 / 60, dtype=approximator.DTYPE)
    z_space = torch.linspace(0, 0.254, dtype=approximator.DTYPE)
    tz_pairs_space = torch.stack((t_space, z_space), dim=-1)
    net_vals = torch.squeeze(approximation.net(tz_pairs_space), dim=1)
    trapz = torch.trapz(net_vals, z_space)
    return trapz


# integrate_matric_potential(1)


def total_water_content_by_mean(trained_model: int, time_in_h: float):
    load_trained_model(trained_model)
    steps = 10000
    t_space = time_in_h * torch.ones(steps, dtype=approximator.DTYPE)
    z_space = torch.linspace(0, 0.254, steps=steps, dtype=approximator.DTYPE)
    tz_pairs_space = torch.stack((t_space, z_space), dim=-1)
    net_vals = approximation.net(tz_pairs_space)
    water_contents = theta_by_h(net_vals)
    mean_water_content = torch.mean(water_contents)
    # mean_water_content = theta_by_h(approximation.use(time_in_h, 0.1))  # @TODO Debug only
    # wsi = mean_water_content * 0.254
    return mean_water_content


total_water_content_by_mean_t0 = total_water_content_by_mean(2, 0)
total_water_content_by_mean_t1 = total_water_content_by_mean(2, 4 / 60)

res_layer_2_theta_t0

res_layer_2_theta_t1

pass
