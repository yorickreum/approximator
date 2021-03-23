# Approximator

Uses PyTorch to approximate arbitrary functions (currently from 2D to 1D) by giving a domain and one or multiple residuals for (parts of) the domain. 

## Examples (in `./examples/`)
### sin
Approximate an explicitly given function.

### circle
Approximate a piecewise given function.

### conservation
Approximate the conservation equation by using its residual.

### laplace
Approximate Laplace's equation by using its residual.
Use StepsDiscretization to ensure that start and endpoints of domain are in the discretized domain, this is important for boundary conditions.
Pretraining is used to only fit the boundary conditions first.
