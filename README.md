# jeig - Eigendecompositions wrapped for jax
[![Continuous integration](https://github.com/mfschubert/jeig/actions/workflows/build-ci.yml/badge.svg)](https://github.com/mfschubert/jeig/actions)
[![PyPI version](https://img.shields.io/pypi/v/jeig)](https://pypi.org/project/jeig/)

## Overview

This package wraps eigendecompositions as provided by jax, cusolver, magma, numpy, scipy, and torch for use with jax. Depending upon your system and your versions of these packages, you may observe significant speed differences. The following were obtained using jax 0.8.0 on a system with 28-core Intel Xeon w7-3465X and NVIDIA RTX4090.

![Speed comparison](https://github.com/mfschubert/jeig/blob/main/docs/speed.png?raw=true)

## Install
jeig can be installed via pip,
```
pip install jeig
```
This will also install torch. If you only need torch for use with jeig, then the CPU-only version could be sufficient and you may wish to install manually as described in the [pytorch docs](https://pytorch.org/get-started/locally/).

## Example usage

```python
import jax
import jeig

matrix = jax.random.normal(jax.random.PRNGKey(0), (1, 2048, 2048)).astype(complex)

%timeit jax.block_until_ready(jeig.eig(matrix, backend="cusolver"))
%timeit jax.block_until_ready(jeig.eig(matrix, backend="lapack"))
%timeit jax.block_until_ready(jeig.eig(matrix, backend="magma"))
%timeit jax.block_until_ready(jeig.eig(matrix, backend="torch"))
```
```
1.31 s ± 43 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
5.44 s ± 379 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
11.1 s ± 937 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
4.93 s ± 92.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

The default torch backend has good performance when performing batched eigendecomposition on many-core CPUs.

```python
matrix = jax.random.normal(jax.random.PRNGKey(0), (8, 2048, 2048)).astype(complex)

%timeit jax.block_until_ready(jeig.eig(matrix, backend="cusolver"))
%timeit jax.block_until_ready(jeig.eig(matrix, backend="lapack"))
%timeit jax.block_until_ready(jeig.eig(matrix, backend="magma"))
%timeit jax.block_until_ready(jeig.eig(matrix, backend="torch"))
```
```
10.4 s ± 116 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
48.1 s ± 6.74 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
1min 33s ± 1.49 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
7.18 s ± 91.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

## Credit
The torch implementation of eigendecomposition is due to a [comment](https://github.com/google/jax/issues/10180#issuecomment-1092098074) by @YouJiacheng.
