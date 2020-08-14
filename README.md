# Approximator

Uses PyTorch to approximate arbitrary functions (currently from 2D to 1D) by giving a domain and one or multiple residuals for (parts of) the domain. 

## Examples
### sin
Approximate a explicitly given function.

### circle
Approximate a piecewise given function.

### laplace
Approximate Laplace's equation by using its residual.
Use StepsDiscretization to ensure that start and endpoints of domain are in the discretized domain, this is important for boundary conditions.

LaTeX-Test: <img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">