# Physics-Informed-Neural-Networks
PINNs are NNs that incorporate physical laws described by DEs into their loss functions to guide the learning process toward solutions that are more consistent with the underlying physics.

# PINNs as scientific inference
In this repo, I will be presenting a set of executable, nine-notebook course that starts with derivatives and reaches research-level questions about conditioning, weak forms, identifiability,
sampling, certification, and scientific validity.


A PINN is best understood as **constrained inference in jet space**:

1. A neural network represents a continuous candidate field
   $u_\theta:\Omega\to\mathbb R^m$.
2. Automatic differentiation lifts every point to its derivative jet
   $J^k u_\theta=(u_\theta,\nabla u_\theta,\ldots,\nabla^k u_\theta)$.
3. A differential equation defines a constraint manifold
   $F(x,J^k u)=0$.
4. Boundary conditions, measurements, conservation laws, and residual probes
   are different scientific instruments observing the same candidate field.
5. Training moves one global field until its lifted graph is compatible with
   those instruments—approximately under penalties or exactly by construction.

This view reveals that a PINN is not merely “a neural network with physics in
the loss.” It is simultaneously a trial-space method, a stochastic weighted-
residual method, a constrained inverse problem, and a differentiable surrogate.

## Primary references

- Raissi, Perdikaris & Karniadakis, “Physics-informed neural networks,”
  *Journal of Computational Physics* 378 (2019),
  [DOI](https://doi.org/10.1016/j.jcp.2018.10.045).
- Wang, Teng & Perdikaris, “Understanding and mitigating gradient pathologies,”
  [arXiv:2001.04536](https://arxiv.org/abs/2001.04536).
- Wang, Yu & Perdikaris, “When and why PINNs fail to train: An NTK perspective,”
  [arXiv:2007.14527](https://arxiv.org/abs/2007.14527).
- Krishnapriyan et al., “Characterizing possible failure modes,”
  [arXiv:2109.01050](https://arxiv.org/abs/2109.01050).
- Wang, Sankaran & Perdikaris, “Respecting causality is all you need,”
  [arXiv:2203.07404](https://arxiv.org/abs/2203.07404).
- Kharazmi, Zhang & Karniadakis, “Variational physics-informed neural networks,”
  [arXiv:1912.00873](https://arxiv.org/abs/1912.00873).
- Sukumar & Srivastava, “Exact imposition of boundary conditions with distance
  functions,” [arXiv:2104.08426](https://arxiv.org/abs/2104.08426).
- Mishra & Molinaro, “Estimates on the generalization error of PINNs,”
  [arXiv:2006.16144](https://arxiv.org/abs/2006.16144).
- De Ryck & Mishra, “Numerical analysis of PINNs and related models,”
  [arXiv:2402.10926](https://arxiv.org/abs/2402.10926).
- Grossmann et al., “Can physics-informed neural networks beat the finite element
  method?”, *IMA Journal of Applied Mathematics* 89 (2024),
  [DOI](https://doi.org/10.1093/imamat/hxae011).


  ## Scope and honest limitation
The provided notebooks contains or will contain are CPU-sized teaching experiments, not performance benchmarks. 
They use smooth one-dimensional or 1+1-dimensional equations so every mathematical and
optimization choice remains inspectable. 
The final notebooks will try to explain how the
same issues extend to systems, irregular geometry, shocks, high dimensions,
operator learning, domain decomposition, and/or uncertainty quantification [hold on until then].
