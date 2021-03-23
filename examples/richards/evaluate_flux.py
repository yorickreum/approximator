import os

import torch

import approximator
from approximator.classes.approximation import Approximation
from approximator.examples.richards.func_lib import theta, K
from approximator.examples.richards.richards import problem, approximation_net, plot_richards_res
from approximator.utils.visualization import plot_approximation, plot_approximation_residuals

approximation = Approximation(
    problem=problem,
    net=approximation_net
)

dir_path = os.path.dirname(os.path.realpath(__file__))
# approximation.load(os.path.join(dir_path, "trained_models/patience-1/net.pt"))
# approximation.load(os.path.join(dir_path, "trained_models/patience-2-400-steps/net.pt"))
approximation.load(os.path.join(dir_path, "results-thesis/example-run/2021-03-19-21-23-53-510723/net.pt"))

mode = "trapz"
n = 1000

# results n = 1000
# mass_balance 100 steps trapz: tensor(1.66777013e+00, grad_fn=<DivBackward0>)
# mass_balance 400 steps trapz: tensor(1.55000981e+00, grad_fn=<DivBackward0>)

# visualize the approximation
# plot_approximation(approximation)
# plot_approximation_residuals(approximation)

x_min, x_max = approximation.problem.domain.x_min, approximation.problem.domain.x_max  # time in hours
y_min, y_max = approximation.problem.domain.y_min, approximation.problem.domain.y_max  # space in meters

# boundaries
all_times = torch.linspace(x_min, x_max, n, requires_grad=True)
lower_boundary = torch.stack((y_min * torch.ones(n), all_times), 1)
higher_boundary = torch.stack((y_max * torch.ones(n), all_times), 1)
all_places = torch.linspace(y_min, y_max, n, requires_grad=True)
initial_boundary = torch.stack((x_min * torch.ones(n), all_places), 1)
final_boundary = torch.stack((x_max * torch.ones(n), all_places), 1)

# calculate net values
higher_boundary_h = approximation.net(higher_boundary)
lower_boundary_h = approximation.net(lower_boundary)

# calculate the fluxes
ones = torch.unsqueeze(torch.ones(n, dtype=approximator.DTYPE, device=approximator.DEVICE), 1)

higher_boundary_dh_d = torch.autograd.grad(
    higher_boundary_h,
    higher_boundary,
    # create_graph=True,
    grad_outputs=ones
)[0]
higher_boundary_dh_dz = higher_boundary_dh_d[:, 1:2]
higher_boundary_K = K(higher_boundary_h)
# flux in m per h (in 3D it would be m^3 per h)
higher_boundary_flux = - higher_boundary_K * (higher_boundary_dh_dz + 1)

lower_boundary_dh_d = torch.autograd.grad(
    lower_boundary_h,
    lower_boundary,
    # create_graph=True,
    grad_outputs=ones
)[0]
lower_boundary_dh_dz = lower_boundary_dh_d[:, 1:2]
lower_boundary_K = K(lower_boundary_h)
# flux in m per h (in 3D it would be m^3 per h)
lower_boundary_flux = - lower_boundary_K * (lower_boundary_dh_dz + 1)

# total_net_flux_water_content by sum
if mode == "sum":
    lower_boundary_flux_sum = torch.sum(lower_boundary_flux)
    higher_boundary_flux_sum = torch.sum(higher_boundary_flux)
    total_net_flux = -higher_boundary_flux_sum + lower_boundary_flux_sum  # in-out --> Zuwachs durch boundaries
    total_net_flux_water = total_net_flux * ((x_max - x_min) / n)
    total_net_flux_water_content = total_net_flux_water / (y_max - y_min)  # <-- theta change by boundaries
# delta_water = sum_i qi * delta_t_i
# *((x_max - x_min) / n)
else:
    # total_net_flux_water_content by trapz
    lower_boundary_water_content = torch.trapz(torch.squeeze(lower_boundary_flux, dim=1), all_times)
    higher_boundary_water_content = torch.trapz(torch.squeeze(higher_boundary_flux, dim=1), all_times)
    total_net_flux_water_content = -higher_boundary_water_content + lower_boundary_water_content

# by sum
if mode == "sum":
    initial_mean_water_content = torch.mean(theta(approximation.net(initial_boundary)))
    final_mean_water_content = torch.mean(theta(approximation.net(final_boundary)))
else:
    # by trapz
    initial_mean_water_content = torch.trapz(torch.squeeze(theta(approximation.net(initial_boundary)), dim=1), all_places)
    final_mean_water_content = torch.trapz(torch.squeeze(theta(approximation.net(final_boundary)), dim=1), all_places)

mean_water_content_diff = final_mean_water_content - initial_mean_water_content
# <-- theta observed in solution, should be equal to change by boundaries

mass_balance = mean_water_content_diff / total_net_flux_water_content
print(mass_balance)

pass