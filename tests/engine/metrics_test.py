# Copyright (c) 2023 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import pymc

from stratcona.engine.metrics import *


def test_lower_quantile_bound():
    # This model sums up to result in a standard normal distribution
    with pymc.Model() as model:
        g1 = pymc.Normal('g1', mu=2, sigma=0.4**0.5)
        g2 = pymc.Normal('g2', mu=-1, sigma=0.4**0.5)
        g3 = pymc.Normal('g3', mu=-1, sigma=0.2**0.5)
        output = pymc.Deterministic('y', g1 + g2 + g3)

    sampler = pymc.compile_pymc([], [output], name='out_sampler', random_seed=456)

    result = worst_case_quantile_credible_interval(sampler, 90)
    assert round(result, 6) == -1.282758  # First 10% quantile of a standard normal distribution is -1.2816 analytically
