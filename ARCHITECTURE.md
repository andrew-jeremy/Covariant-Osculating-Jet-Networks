# Architectural Notes

## 1. Metric-routed contacts

Each contact center `mu_c` has a positive metric

`G_c = beta_c I + M_c M_c^T`, with `beta_c = softplus(theta_c) + eps`.

This guarantees positive-definite routing metrics without a full `D x D` matrix.

## 2. Signed second-order geometry

For every contact/output pair,

`H = U diag(lambda) U^T + alpha I`.

Because `lambda` is signed, the layer can represent saddle-like curvature in arbitrary learned directions. Direction tensors are normalized column-wise during evaluation.

## 3. Third-order variation

A symmetric cubic tensor is represented by

`T[delta,delta,delta] = sum_s a_s (t_s^T delta)^3`.

This captures curvature variation without storing a `D x D x D` tensor.

## 4. Differential gluing

For contact pair `(c,d)`, samples are weighted by detached overlap `gamma_c gamma_d`. The penalty may include:

- C0: squared value mismatch.
- C1: squared Jacobian mismatch.
- C2: Hessian mismatch estimated with Rademacher probes `z`, using `E ||(H_c-H_d)z||^2 = ||H_c-H_d||_F^2`.

## 5. What "covariant" means here

The implementation supports exact covariance under orthogonal reparameterizations of the latent coordinates. It is not a general-coordinate differential-geometric connection and does not implement arbitrary diffeomorphism invariance or transition-map cocycles.
