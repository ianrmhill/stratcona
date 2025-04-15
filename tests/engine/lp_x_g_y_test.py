# Copyright (c) 2025 Ian Hill
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import jax.random as rand
import numpyro.distributions as dists

import stratcona
from stratcona.engine.inference import int_out_v
from stratcona.modelling.relmodel import TestDef


def test_basic_no_latents():
    mb = stratcona.SPMBuilder('faux_hierarchical')
    mb.add_hyperlatent('x', dists.Bernoulli, {'probs': 0.5})
    square = lambda x: x**2
    mb.add_intermediate('x2', square)
    mb.add_params(noise=0.01)
    mb.add_observed('y', dists.Normal, {'loc': 'x2', 'scale': 'noise'}, 1)

    am = stratcona.AnalysisManager(mb.build_model(), rng_seed=943837)
    d = TestDef('arb', {'1': {'lot': 1, 'chp': 1}}, {'1': {'temp': 50}})

    k1 = rand.key(27834)
    x_s = {'x': jnp.array([0, 1])}
    y_s = {'1_y': jnp.array([0, 1])}
    lp_x_g_y, stats = int_out_v(k1, am.relmdl, (2, 1, 2), d.dims, d.conds, x_s, y_s, y_noise={'y': 0.01})
    assert lp_x_g_y[0][0] > 1 and lp_x_g_y[1][1] > 1
    assert lp_x_g_y[0][1] < 0 and lp_x_g_y[1][0] < 0