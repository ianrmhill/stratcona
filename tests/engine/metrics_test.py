# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pymc
import pytensor as pt

from stratcona.engine.metrics import *


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
