# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpyro.distributions as dist

from stratcona.modelling.builder import _HyperLatent


def test_hyperlatent_variable():
    x = _HyperLatent('x', dist.InverseGamma, {'concentration': 3, 'rate': 5})
    assert x.dist == dist.InverseGamma
    assert x.is_continuous is True
    assert round(x.compute_prior_entropy(), 2) == 1.61
    assert round(x.get_dist_variance(), 2) == 6.25
