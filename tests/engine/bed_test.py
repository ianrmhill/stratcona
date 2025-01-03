# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import jax.random as rand
import numpyro.distributions as dist

from stratcona.engine.bed import entropy


def test_entropy():
    # 1) Test differential entropy calculation for a normal distribution
    v = 1.2
    d = dist.Normal(0, v)
    # Entropy of a normal distribution with variance v^2
    d_entropy_analytic = (0.5*jnp.log(2*jnp.pi*(v**2))) + 0.5
    k = rand.key(92347)
    s = d.sample(k, (100_000,))
    d_entropy = entropy(s, d.log_prob)
    assert jnp.round(d_entropy, 2) == jnp.round(d_entropy_analytic, 2)

    # 2) Test differential entropy calculation for a uniform distribution
    a, b = -5, 5
    u = dist.Uniform(a, b)
    u_entropy_analytic = -jnp.log(1 / (b - a))
    k = rand.key(10473)
    s = u.sample(k, (100_000,))
    u_entropy = entropy(s, u.log_prob)
    assert jnp.round(u_entropy, 2) == jnp.round(u_entropy_analytic, 2)

    # 2) Test LDDP entropy calculation for a normal distribution
    c = jnp.log(2**23)
    d_lddp_analytic = d_entropy_analytic - u_entropy_analytic + c
    k = rand.key(63916)
    s = d.sample(k, (100_000,))
    d_lddp = entropy(s, d.log_prob, limiting_density_range=(-5, 5))
    assert jnp.round(d_lddp, 2) == jnp.round(d_lddp_analytic, 2)
