# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import jax.random as rand

from stratcona.engine.metrics import *
from stratcona.engine.bed import qx_hdcr_width


def test_lower_quantile_bound():
    # This model sums up to result in a standard normal distribution
    with pymc.Model() as model:
        g1 = pymc.Normal('g1', mu=2, sigma=0.4**0.5)
        g2 = pymc.Normal('g2', mu=-1, sigma=0.4**0.5)
        g3 = pymc.Normal('g3', mu=-1, sigma=0.2**0.5)
        output = pymc.Deterministic('y', g1 + g2 + g3)

    sampler = pymc.compile_pymc([], [output], name='out_sampler', random_seed=456)

    result = worst_case_quantile_credible_region(sampler, 90, 30000)
    assert round(result, 6) == -1.282758  # First 10% quantile of a standard normal distribution is -1.2816 analytically


def test_highest_density_region():
    with pymc.Model() as model:
        g1 = pymc.Normal('g1', mu=2, sigma=0.3)
        g2 = pymc.Normal('g2', mu=3, sigma=0.3)
        sel = pymc.Bernoulli('sel', p=0.5)
        mix = pt.tensor.where(pt.tensor.eq(sel, 0), g1, g2)
        output = pymc.Deterministic('y', mix)

    sampler = pymc.compile_pymc([], [output], name='out_sampler', random_seed=654)
    result = highest_density_credible_region(sampler, 50, 30000, 100)
    assert np.allclose(np.round(result, 4), [[1.7895, 2.2153], [2.8049, 3.1652]])


def test_qxhdcr_width():
    q = 0.9
    n_bins = 100
    k1, k2 = rand.split(rand.key(482394))
    z = rand.normal(k1, (3, 1000))
    w = rand.uniform(k2, (3, 4))
    # With uniform random weights, the regions should all be similar
    widths = qx_hdcr_width(z, w, q, n_bins)
    assert jnp.allclose(jnp.round(widths, 2), jnp.array([3.14, 3.14, 3.14, 3.21]))
    # With weights prioritizing different normal distributions of different widths, we can predict relative region sizes
    z = z.at[0, :].mul(0.5)
    z = z.at[2, :].mul(2.5)
    w = jnp.array([[0.3, 0.9, 0.2, 0.01], [0.3, 0.1, 0.5, 0.98], [0.3, 0.1, 0.9, 0.01]])
    widths = qx_hdcr_width(z, w, q, n_bins)
    assert jnp.allclose(jnp.round(widths, 2), jnp.array([5.17, 2.13, 6.54, 3.35]))
    # Shifting the distributions increases the overall spread of the distribution, increasing region sizes
    z = z.at[0, :].add(4)
    z = z.at[1, :].add(-4)
    widths = qx_hdcr_width(z, w, q, n_bins)
    assert jnp.allclose(jnp.round(widths, 2), jnp.array([8.82, 4.72, 9.13, 3.50]))
