# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import jax.random as rand
from jax.scipy.special import erfinv
import numpyro.distributions as dist

from stratcona.engine.bed import weighted_quantile


def test_importance_weighted_quantiles():
    def normal_q(u, v, q):
        return (jnp.sqrt(2) * v * erfinv((2 * q) - 1)) + u

    u, v, quantile = 0.0, 1.2, 0.25
    d = dist.Normal(u, v)
    # Quantile function of a normal distribution with variance v^2
    quantile_analytic = round(float(normal_q(u, v, quantile)), 2)

    k = rand.key(64983)
    k1, k2, k3 = rand.split(k, 3)

    # Sampling-based estimate done via IS with q(x)=p(x)
    s1 = d.sample(k1, (1_000_000,))
    w1 = jnp.ones_like(s1)
    m = weighted_quantile(s1, w1, quantile)
    assert round(m, 2) == quantile_analytic

    # Via IS with uniform q(x)
    q_u = dist.Uniform(-10, 10)
    s2 = q_u.sample(k2, (1_000_000,))
    w2 = jnp.exp(d.log_prob(s2) - q_u.log_prob(s2))
    m = weighted_quantile(s2, w2, quantile)
    assert round(m, 2) == quantile_analytic

    # Via IS with a wider normal q(x)
    q_n = dist.Normal(-2, 5)
    s3 = q_n.sample(k3, (1_000_000,))
    w3 = jnp.exp(d.log_prob(s3) - q_n.log_prob(s3))
    m = weighted_quantile(s3, w3, quantile)
    assert round(m, 2) == quantile_analytic
