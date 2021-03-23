# Approximator

Uses PyTorch to approximate arbitrary functions (currently from 2D to 1D) by giving a domain and one or multiple residuals for (parts of) the domain.
Can be used especially for the approximation of partial differential equations.

## Examples (in `./examples/sin`)
### Sinus
Approximate an explicitly given function.

### Circle
Approximate a piecewise given function.

### Conservation
Approximate the solution of an initial-boundary value problem governed by the conservation equation by using its residual.

### Laplace
Approximate a solution of an initial-boundary value problem governed by the Laplace's equation.
Use StepsDiscretization to ensure that start and endpoints of domain are in the discretized domain, this is important for boundary conditions.
Pretraining is used to only fit the boundary conditions first.

[comment]: <> (LaTeX-Test: <img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">)

### Heat
Approximate the heat equation.
![Heat equation approximation](examples/heat/results-thesis/example-run/figs/plt_Approximated_solution_of_the_heat_equation.png)

### Richardson-Richards equation
Approximate the Richardson-Richards equation in the setup [described by Michael A. Celia, Efthimios T. Bouloutas and Rebecca L. Zarba in 1990](https://doi.org/10.1029/WR026i007p01483).
![RRE approximation](examples/richards/results-thesis/example-run/figs/2021-03-19-21-23-53-510723/plt_Approximated_solution_for_h_cm.png)