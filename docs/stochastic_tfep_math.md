# Stochastic Path-Weighted TFEP Mathematics

This note defines the experimental stochastic TFEP estimator. It is not the deterministic production TFEP estimator and must not reuse the deterministic work formula when stochastic kernels are active.

## Deterministic TFEP Recap

The current production map is deterministic and invertible:

```text
x_B = M_theta(x_A)
logJ = log |det dM_theta/dx_A|
```

The existing reduced work convention is:

```text
w_A_to_B = u_B(M_theta(x_A)) - u_A(x_A) - logJ
```

This is the default and remains unchanged.

## Forward Path

A stochastic path is:

```text
x_0 ~ A

y_k = f_k(x_k)
x_{k+1} ~ K_k^F(. | y_k)
```

The path is:

```text
gamma_F = (x_0, y_0, x_1, ..., y_{K-1}, x_K)
```

The forward conditional path probability is:

```text
Q_F(gamma_F | x_0) = product_k delta(y_k - f_k(x_k)) K_k^F(x_{k+1} | y_k)
```

## Reverse Protocol

The selected reverse protocol is stochastic reverse first, then inverse deterministic map:

```text
x_K ~ B

for k = K-1 ... 0:
    y_k ~ K_k^R(. | x_{k+1})
    x_k = f_k^{-1}(y_k)
```

The reverse conditional probability is:

```text
Q_R(gamma_R | x_K) = product_k K_k^R(y_k | x_{k+1}) delta(x_k - f_k^{-1}(y_k))
```

The deterministic delta-ratio gives the same Jacobian sign as deterministic TFEP.

## Path Work

Define:

```text
Delta f_AB = -log(Z_B / Z_A)
```

The forward path work is the negative log path-density ratio:

```text
w_F(gamma) = -log [ exp(-u_B(x_K)) Q_R(gamma_R | x_K)
                    / exp(-u_A(x_0)) Q_F(gamma_F | x_0) ]
```

Expanded:

```text
w_F = u_B(x_K)
    - u_A(x_0)
    - sum_k log |det df_k/dx_k|
    + sum_k log K_k^F(x_{k+1} | y_k)
    - sum_k log K_k^R(y_k | x_{k+1})
```

The signs of the stochastic density terms are therefore `+logq_forward - logq_reverse` with this convention.

## Jarzynski Identity

```text
exp(-Delta f_AB) = E_{A,Q_F}[exp(-w_F)]
```

The forward one-sided estimator is:

```text
Delta f_AB = -log mean(exp(-w_F))
```

## Reverse Work And BAR

For reverse paths sampled from B to A, the same generic work formula is used with source B and target A. The result is stored as `w10`, matching the existing TFEP convention:

```text
w10 = u_A(x_0)
    - u_B(x_K)
    - sum_k logJ_inverse_k
    + sum_k log K_k^R(y_k | x_{k+1})
    - sum_k log K_k^F(x_{k+1} | y_k)
```

These `w01` and `w10` arrays can be passed to the existing BAR objective only after the sign-convention tests pass.

## Deterministic Limit

With no stochastic kernels:

```text
sum logq_forward = 0
sum logq_reverse = 0
sum logJ = log |det dM/dx|
```

Then:

```text
w_F = u_B(M(x_A)) - u_A(x_A) - log |det dM/dx_A|
```

This must equal the current deterministic TFEP work numerically.
