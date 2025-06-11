# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import jax.random as rand
import numpyro.distributions as dist

from stratcona.engine.bed import entropy


def test_entropy():
    def normal_entropy(variance):
        return (0.5*jnp.log(2*jnp.pi*(variance**2))) + 0.5

    # 1) Test differential entropy calculation for a normal distribution
    v = 1.2
    d = dist.Normal(0, v)
    # Entropy of a normal distribution with variance v^2
    d_entropy_analytic = normal_entropy(v)
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

    # 3) Test LDDP entropy calculation for a normal distribution
    c = jnp.log(2**23)
    d_lddp_analytic = d_entropy_analytic - u_entropy_analytic + c
    k = rand.key(63916)
    s = d.sample(k, (100_000,))
    d_lddp = entropy(s, d.log_prob, limiting_density_range=(-5, 5))
    assert jnp.round(d_lddp, 2) == jnp.round(d_lddp_analytic, 2)

    # 4) Test LDDP composition
    v2 = 1.4
    e = dist.Normal(0, v2)
    k = rand.key(98274)
    s2 = d.sample(k, (100_000,))
    e_lddp = entropy(s2, e.log_prob, limiting_density_range=(-5, 5))

    def ed_lp(samples):
        return d.log_prob(samples['d']) + e.log_prob(samples['e'])
    lddp_additive = e_lddp + d_lddp
    joint_lddp = entropy({'d': s, 'e': s2}, ed_lp, limiting_density_range=(-5, 5), d=2)
    assert jnp.round(lddp_additive, 2) == jnp.round(joint_lddp, 2)

    # 5) Test LDDP out of range bounds handling
    e_bad_lddp = entropy(s2, e.log_prob, limiting_density_range=(-2, 2))
    assert e_bad_lddp > e_lddp

    # 6) Test zero entropy LDDP conditions
    g = dist.Uniform(1.0, 1.0001)
    k = rand.key(58365)
    s3 = g.sample(k, (100_000,))
    g_lddp = entropy(s3, g.log_prob, limiting_density_range=(-838.8608, 838.8607))
    assert jnp.round(g_lddp, 4) == 0.0
    g_lddp_2 = entropy(s3, g.log_prob, limiting_density_range=(1.0, 1.128), precision=float(2**7))
    assert jnp.round(g_lddp_2, 4) == 0.0
