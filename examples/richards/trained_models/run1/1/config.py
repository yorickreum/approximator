problem = Problem(
    Domain(
        x_min=0,  # x is time t in s
        x_max=20,
        y_min=0,  # y is elevation z in cm
        y_max=40  # attention: in SimPEG this is probably 39?
    ),
    [
        Constraint(
            lambda x, y: x == 0,
            lambda x, y, prediction: (prediction - (1.02 * y - 61.5)) ** 2  # linear starting conditions
        ),
        Constraint(
            lambda x, y: y == 40,
            lambda x, y, prediction: (prediction + 20.7) ** 2
        ),
        Constraint(
            lambda x, y: y == 0,
            lambda x, y, prediction: (prediction + 61.5) ** 2
        ),
        Constraint(
            lambda x, y: not (x == 0 or y == 40 or y == 0),
            lambda input, prediction: get_res(input, prediction) ** 2
        )
    ]
)

approximation = Approximation(
    problem=problem,
    discretization=StepsDiscretization(
        x_steps=400,
        y_steps=200
    ),
    n_hidden_layers=4,
    n_neurons_per_layer=10,
    learning_rate=.001,
    epochs=int(1e4)
)