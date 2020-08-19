# from approximator.examples.sin import plot_sin
# plot_sin()

# from approximator.examples.circle import plot_circle
# plot_circle()

# from approximator.examples.laplace import plot_laplace
# plot_laplace()

# from approximator.examples.conservation import plot_conservation
# plot_conservation()

# from approximator.examples.richards.richards import train_richards, plot_richards
# plot_richards(train=False)
# train_richards()

from approximator.examples.heat.heat import train_heat, plot_heat, plot_heat_res, plot_heat_difference

last_model = 5
# train_heat(last_model)
plot_heat(last_model)
plot_heat_res(last_model)
plot_heat_difference(last_model)
