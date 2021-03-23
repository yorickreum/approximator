# Approximator

A tool to use neural networks to approximate arbitrary functions (currently ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cmathbb%7BR%7D%5E2%20%5Cto%20%5Cmathbb%7BR%7D)) by specifying a domain and one or multiple residuals for (parts of) the domain.
Can be used especially for the approximation of solutions of partial differential equations.
Based on [PyTorch](https://github.com/pytorch/pytorch).

## Examples (in `examples`)
### Sinus
Approximate an explicitly given function.

### Circle
Approximate a piecewise given function.

### Conservation
Approximate the solution of an initial-boundary value problem governed by the conservation equation by using its residual.

### Laplace
Approximate a solution of an initial-boundary value problem governed by Laplace's equation:  
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%7B%5Cdisplaystyle%20%5Cnabla%20%5E%7B2%7Du%3D%7B%5Cfrac%20%7B%5Cpartial%20%5E%7B2%7Du%7D%7B%5Cpartial%20x%5E%7B2%7D%7D%7D&plus;%7B%5Cfrac%20%7B%5Cpartial%20%5E%7B2%7Du%7D%7B%5Cpartial%20y%5E%7B2%7D%7D%7D%3D0%7D)

Uses `StepsDiscretization` to ensure that start and endpoints of domain are in the discretized domain, this is important for boundary conditions.
Pretraining is used to first fit the neural network to the boundary conditions, then to the PDE.

### Heat
Approximate the heat equation:
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%7B%5Cdisplaystyle%20%7B%5Cfrac%20%7B%5Cpartial%20u%7D%7B%5Cpartial%20t%7D%7D%3D%5Calpha%20%5Cleft%28%7B%5Cfrac%20%7B%5Cpartial%20%5E%7B2%7Du%7D%7B%5Cpartial%20x%5E%7B2%7D%7D%7D&plus;%7B%5Cfrac%20%7B%5Cpartial%20%5E%7B2%7Du%7D%7B%5Cpartial%20y%5E%7B2%7D%7D%7D%5Cright%29%7D)

![Heat equation approximation](examples/heat/results-thesis/example-run/figs/plt_Approximated_solution_of_the_heat_equation.png)

### Richardson-Richards equation
Approximate the Richardson-Richards equation in the setup [described by Michael A. Celia, Efthimios T. Bouloutas and Rebecca L. Zarba in 1990](https://doi.org/10.1029/WR026i007p01483).

The RRE:  
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cfrac%7B%5Cpartial%20%5CTheta%7D%7B%5Cpartial%20t%7D%20-%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20z%7D%20%5Cleft%28%20D%28%5CTheta%29%29%20%5Cfrac%7B%5Cpartial%20%5CTheta%7D%7B%5Cpartial%20z%7D%20%5Cright%29%20-%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20z%7D%20K%28%5CTheta%29%20%3D%200)  
with constitutive equation:  
![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cbegin%7Balign*%7D%20%5CTheta%28h%29%20%26%3D%20%5Cfrac%7B%5Calpha%20%5Cleft%28%20%5CTheta_s%20-%20%5CTheta_r%20%5Cright%29%7D%7B%5Calpha%20&plus;%20%7Ch%7C%5E%5Cbeta%7D%20&plus;%20%5CTheta_r%20%5C%5C%20K%28h%29%20%26%3D%20K_s%20%5Cfrac%7BA%7D%7BA%20&plus;%20%7Ch%7C%5E%5Cgamma%7D%20%5Cend%7Balign*%7D)

![RRE approximation](examples/richards/results-thesis/example-run/figs/2021-03-19-21-23-53-510723/plt_Approximated_solution_for_h_cm.png)