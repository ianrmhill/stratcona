# Copyright (c) 2024 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import numpyro.distributions as dists
from numpyro.handlers import seed, trace
import jax.random as rand
import jax.numpy as jnp

from stratcona.modelling.builder import _HyperLatent, SPMBuilder


def test_hyperlatent_variable():
    x = _HyperLatent('x', dists.InverseGamma, {'concentration': 3, 'rate': 5})
    assert x.dist_type == 'continuous'
    assert round(x.compute_prior_entropy(), 2) == 1.61
    assert round(x.get_dist_variance(), 2) == 6.25


def test_build_model():
    builder = SPMBuilder('test')
    assert builder.model_name == 'test'

    builder.add_hyperlatent('x_c', dists.Normal, {'loc': 0, 'scale': 0})
    builder.add_hyperlatent('x_r', dists.Normal, {'loc': 0, 'scale': 0})
    builder.add_hyperlatent('x_chp', dists.HalfCauchy, {'scale': 0})
    builder.add_latent('x', dists.Gamma, {'concentration': 'x_c', 'rate': 'x_r'}, chp_var='x_chp')

    builder.add_dependent('nx', lambda x, n: n * x)
    builder.add_measured('y', dists.Normal, {'loc': 'nx', 'scale': 'meas_error'})

    mdl = builder.build_model()
    rng = rand.key(89)
    mdl_ts = trace(seed(mdl, rng))
    tr = mdl_ts.get_trace({'meas_error': 0.1}, {'e1': {'lot': 2, 'chp': 3, 'y': 4}}, {'e1': {'n': 2}},
                          {'x_c': {'loc': 12, 'scale': 0.5}, 'x_r': {'loc': 3, 'scale': 0.2}, 'x_chp': {'scale': 0.2}})
    print(tr)
